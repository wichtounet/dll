//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/neural_layer.hpp"

namespace dll {

/*!
 * \brief Standard dense layer of neural network.
 */
template <typename Desc>
struct dyn_dense_layer final : neural_layer<dyn_dense_layer<Desc>, Desc> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = dyn_dense_layer<desc>;
    using base_type = neural_layer<this_type, desc>;

    static constexpr auto activation_function = desc::activation_function;
    static constexpr auto w_initializer       = desc::w_initializer;
    static constexpr auto b_initializer       = desc::b_initializer;

    using input_one_t  = etl::dyn_matrix<weight, 1>;
    using output_one_t = etl::dyn_matrix<weight, 1>;
    using input_t      = std::vector<input_one_t>;
    using output_t     = std::vector<output_one_t>;

    using w_type = etl::dyn_matrix<weight, 2>;
    using b_type = etl::dyn_matrix<weight, 1>;

    //Weights and biases
    w_type w; //!< Weights
    b_type b; //!< Hidden biases

    //Backup Weights and biases
    std::unique_ptr<w_type> bak_w; //!< Backup Weights
    std::unique_ptr<b_type> bak_b; //!< Backup Hidden biases

    size_t num_visible;
    size_t num_hidden;

    dyn_dense_layer() : base_type() {}

    void init_layer(size_t nv, size_t nh) {
        num_visible = nv;
        num_hidden  = nh;

        w = etl::dyn_matrix<weight, 2>(num_visible, num_hidden);
        b = etl::dyn_matrix<weight, 1>(num_hidden);

        initializer_function<w_initializer>::initialize(w, input_size(), output_size());
        initializer_function<b_initializer>::initialize(b, input_size(), output_size());
    }

    /*!
     * \brief Returns the input size of this layer
     */
    size_t input_size() const noexcept {
        return num_visible;
    }

    /*!
     * \brief Returns the output size of this layer
     */
    size_t output_size() const noexcept {
        return num_hidden;
    }

    /*!
     * \brief Returns the number of parameters of this layer
     */
    size_t parameters() const noexcept {
        return num_visible * num_hidden;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_short_string() const {
        char buffer[1024];
        snprintf(buffer, 1024, "Dense: %lu -> %s -> %lu", num_visible, to_string(activation_function).c_str(), num_hidden);
        return {buffer};
    }

    using base_type::activate_hidden;

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() == 1)>
    void activate_hidden(H&& output, const V& v) const {
        output = f_activate<activation_function>(b + v * w);
    }

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() != 1)>
    void activate_hidden(H&& output, const V& v) const {
        output = f_activate<activation_function>(b + etl::reshape(v, num_visible) * w);
    }

    template <typename V>
    auto batch_activate_hidden(const V& v) const {
        const auto Batch = etl::dim<0>(v);

        etl::dyn_matrix<weight, 2> output(Batch, num_hidden);
        batch_activate_hidden(output, v);
        return output;
    }

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() == 2)>
    void batch_activate_hidden(H&& output, const V& v) const {
        const auto Batch = etl::dim<0>(v);

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        output = v * w;

        if (activation_function == function::SOFTMAX) {
            output = bias_add_2d(output, b);

            for (size_t i = 0; i < Batch; ++i) {
                output(i) = f_activate<activation_function>(output(i));
            }
        } else {
            output = f_activate<activation_function>(bias_add_2d(output, b));
        }
    }

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() != 2)>
    void batch_activate_hidden(H&& output, const V& input) const {
        auto Batch = etl::dim<0>(input);

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        output = etl::reshape(input, Batch, num_visible) * w;

        if (activation_function == function::SOFTMAX) {
            output = bias_add_2d(output, b);

            for (size_t i = 0; i < Batch; ++i) {
                output(i) = f_activate<activation_function>(output(i));
            }
        } else {
            output = f_activate<activation_function>(bias_add_2d(output, b));
        }
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    output_one_t prepare_one_output() const {
        return output_one_t(num_hidden);
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    output_t prepare_output(size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(size_t i = 0; i < samples; ++i){
            output.emplace_back(num_hidden);
        }
        return output;
    }

    void prepare_input(input_one_t& input) const {
        input = input_one_t(num_visible);
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
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
        context.errors = f_derivative<activation_function>(context.output) >> context.errors;
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        // The reshape has no overhead, so better than SFINAE for nothing
        auto batch_size = etl::dim<0>(output);
        etl::reshape(output, batch_size, num_visible) = context.errors * etl::transpose(w);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        context.w_grad = batch_outer(context.input, context.errors);
        context.b_grad = bias_batch_sum_2d(context.errors);
    }
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<dyn_dense_layer<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = true;  ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = false; ///< Indicates if the layer is a patches layer
    static constexpr bool is_dynamic    = true;  ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dyn_dense_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_dense_layer<Desc>, L> {
    using layer_t = dyn_dense_layer<Desc>;
    using weight  = typename layer_t::weight;

    static constexpr auto batch_size = DBN::batch_size;

    etl::dyn_matrix<weight, 2> w_grad;
    etl::dyn_matrix<weight, 1> b_grad;

    etl::dyn_matrix<weight, 2> w_inc;
    etl::dyn_matrix<weight, 1> b_inc;

    etl::dyn_matrix<weight, 2> input;
    etl::dyn_matrix<weight, 2> output;
    etl::dyn_matrix<weight, 2> errors;

    sgd_context(layer_t& layer)
            : w_grad(layer.num_visible, layer.num_hidden), b_grad(layer.num_hidden),
              w_inc(layer.num_visible, layer.num_hidden, 0.0), b_inc(layer.num_hidden, 0.0),
              input(batch_size, layer.num_visible, 0.0), output(batch_size, layer.num_hidden, 0.0), errors(batch_size, layer.num_hidden, 0.0) {}
};


} //end of dll namespace
