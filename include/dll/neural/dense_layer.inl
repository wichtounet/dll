//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/neural_layer.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Standard dense layer of neural network.
 */
template <typename Desc>
struct dense_layer final : neural_layer<dense_layer<Desc>, Desc> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = dense_layer<desc>;
    using base_type = neural_layer<this_type, desc>;

    static constexpr const std::size_t num_visible = desc::num_visible;
    static constexpr const std::size_t num_hidden  = desc::num_hidden;

    static constexpr auto activation_function = desc::activation_function;
    static constexpr auto w_initializer       = desc::w_initializer;
    static constexpr auto b_initializer       = desc::b_initializer;

    using input_one_t  = etl::fast_dyn_matrix<weight, num_visible>;
    using output_one_t = etl::fast_dyn_matrix<weight, num_hidden>;
    using input_t      = std::vector<input_one_t>;
    using output_t     = std::vector<output_one_t>;

    using w_type = etl::fast_matrix<weight, num_visible, num_hidden>;
    using b_type = etl::fast_matrix<weight, num_hidden>;

    //Weights and biases
    w_type w; //!< Weights
    b_type b; //!< Hidden biases

    //Backup Weights and biases
    std::unique_ptr<w_type> bak_w; //!< Backup Weights
    std::unique_ptr<b_type> bak_b; //!< Backup Hidden biases

    /*!
     * \brief Initialize a dense layer with basic weights.
     *
     * The weights are initialized from a normal distribution of
     * zero-mean and unit variance.
     */
    dense_layer() : base_type() {
        initializer_function<w_initializer>::initialize(w, input_size(), output_size());
        initializer_function<b_initializer>::initialize(b, input_size(), output_size());
    }

    /*!
     * \brief Returns the input size of this layer
     */
    static constexpr std::size_t input_size() noexcept {
        return num_visible;
    }

    /*!
     * \brief Returns the output size of this layer
     */
    static constexpr std::size_t output_size() noexcept {
        return num_hidden;
    }

    /*!
     * \brief Returns the number of parameters of this layer
     */
    static constexpr std::size_t parameters() noexcept {
        return num_visible * num_hidden;
    }

    static std::string to_short_string() {
        char buffer[1024];
        snprintf(buffer, 1024, "Dense: %lu -> %s -> %lu", num_visible, to_string(activation_function).c_str(), num_hidden);
        return {buffer};
    }

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() == 1)>
    void activate_hidden(H&& output, const V& v) const {
        dll::auto_timer timer("dense:activate_hidden");

        output = f_activate<activation_function>(b + v * w);
    }

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() != 1)>
    void activate_hidden(H&& output, const V& v) const {
        dll::auto_timer timer("dense:activate_hidden");

        output = f_activate<activation_function>(b + etl::reshape<num_visible>(v) * w);
    }

    template <typename V, cpp_enable_if((etl::decay_traits<V>::is_fast))>
    auto batch_activate_hidden(const V& v) const {
        static constexpr auto Batch = etl::decay_traits<V>::template dim<0>();

        etl::fast_dyn_matrix<weight, Batch, num_hidden> output;
        batch_activate_hidden(output, v);
        return output;
    }

    template <typename V, cpp_disable_if((etl::decay_traits<V>::is_fast))>
    auto batch_activate_hidden(const V& v) const {
        const auto Batch = etl::dim<0>(v);

        etl::dyn_matrix<weight, 2> output(Batch, num_hidden);
        batch_activate_hidden(output, v);
        return output;
    }

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() == 2)>
    void batch_activate_hidden(H&& output, const V& v) const {
        dll::auto_timer timer("dense:batch_activate_hidden");

        const auto Batch = etl::dim<0>(v);

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        if (activation_function == function::SOFTMAX) {
            auto expr = etl::force_temporary(etl::rep_l(b, Batch) + v * w);

            for (std::size_t i = 0; i < Batch; ++i) {
                output(i) = f_activate<activation_function>(expr(i));
            }
        } else {
            // The temporary allows vectorization of the activation function -> speedups
            auto expr = etl::force_temporary(etl::rep_l(b, Batch) + v * w);
            output = f_activate<activation_function>(expr);
        }
    }

    //TODO Find a way to reduce this duplication without too much
    //overhead

    template <typename H, typename V, cpp_enable_if((etl::decay_traits<V>::dimensions() != 2 && etl::decay_traits<V>::is_fast))>
    void batch_activate_hidden(H&& output, const V& input) const {
        dll::auto_timer timer("dense:batch_activate_hidden");

        static constexpr auto Batch = etl::decay_traits<V>::template dim<0>();

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        if (activation_function == function::SOFTMAX) {
            auto expr = etl::force_temporary(etl::rep_l(b, Batch) + etl::reshape<Batch, num_visible>(input) * w);

            for (std::size_t i = 0; i < Batch; ++i) {
                output(i) = f_activate<activation_function>(expr(i));
            }
        } else {
            // The temporary allows vectorization of the activation function -> speedups
            auto expr = etl::force_temporary(etl::rep_l(b, Batch) + etl::reshape<Batch, num_visible>(input) * w);
            output = f_activate<activation_function>(expr);
        }
    }

    template <typename H, typename V, cpp_enable_if((etl::decay_traits<V>::dimensions() != 2 && !etl::decay_traits<V>::is_fast))>
    void batch_activate_hidden(H&& output, const V& input) const {
        dll::auto_timer timer("dense:batch_activate_hidden");

        const auto Batch = etl::dim<0>(input);

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        if (activation_function == function::SOFTMAX) {
            auto expr = etl::force_temporary(etl::rep_l(b, Batch) + etl::reshape(input, Batch, num_visible) * w);

            for (std::size_t i = 0; i < Batch; ++i) {
                output(i) = f_activate<activation_function>(expr(i));
            }
        } else {
            // The temporary allows vectorization of the activation function -> speedups
            auto expr = etl::force_temporary(etl::rep_l(b, Batch) + etl::reshape(input, Batch, num_visible) * w);
            output = f_activate<activation_function>(expr);
        }
    }

    template <typename Input>
    output_one_t prepare_one_output() const {
        return {};
    }

    template <typename Input>
    static output_t prepare_output(std::size_t samples) {
        return output_t{samples};
    }

    template<typename DLayer>
    static void dyn_init(DLayer& dyn){
        dyn.init_layer(num_visible, num_hidden);
    }

    /*!
     * \brief Adapt the errors, called before backpropagation of the errors.
     *
     * This must be used by layers that have both an activation fnction and a non-linearity.
     *
     * \param context the training context
     */
    template<typename C>
    void adapt_errors(C& context) const {
        dll::auto_timer timer("dense:adapt_errors");

        if(activation_function != function::IDENTITY){
            context.errors = f_derivative<activation_function>(context.output) >> context.errors;
        }
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        dll::auto_timer timer("dense:backward_batch");

        // The reshape has no overhead, so better than SFINAE for nothing
        constexpr const auto Batch = etl::decay_traits<decltype(context.errors)>::template dim<0>();
        etl::reshape<Batch, num_visible>(output) = context.errors * etl::transpose(w);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        dll::auto_timer timer("dense:compute_gradients");

        context.w_grad = batch_outer(context.input, context.errors);
        context.b_grad = etl::sum_l(context.errors);
    }
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const std::size_t dense_layer<Desc>::num_visible;

template <typename Desc>
const std::size_t dense_layer<Desc>::num_hidden;

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<dense_layer<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = true;  ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false;  ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = false; ///< Indicates if the layer is a patches layer
    static constexpr bool is_augment    = false; ///< Indicates if the layer is an augment layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief specialization of sgd_context for dense_layer
 */
template <typename DBN, typename Desc>
struct sgd_context<DBN, dense_layer<Desc>> {
    using layer_t = dense_layer<Desc>;
    using weight  = typename layer_t::weight;

    static constexpr auto num_visible = layer_t::num_visible;
    static constexpr auto num_hidden  = layer_t::num_hidden;

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, num_visible, num_hidden> w_grad;
    etl::fast_matrix<weight, num_hidden> b_grad;

    etl::fast_matrix<weight, num_visible, num_hidden> w_inc;
    etl::fast_matrix<weight, num_hidden> b_inc;

    etl::fast_matrix<weight, batch_size, num_visible> input;
    etl::fast_matrix<weight, batch_size, num_hidden> output;
    etl::fast_matrix<weight, batch_size, num_hidden> errors;

    sgd_context()
            : w_inc(0.0), b_inc(0.0), output(0.0), errors(0.0) {}
};

} //end of dll namespace
