//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
struct dense_layer_impl final : neural_layer<dense_layer_impl<Desc>, Desc> {
    using desc        = Desc;                          ///< The descriptor of the layer
    using weight      = typename desc::weight;         ///< The data type for this layer
    using this_type   = dense_layer_impl<desc>;        ///< The type of this layer
    using base_type   = neural_layer<this_type, desc>; ///< The base type
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    static inline constexpr size_t num_visible = desc::num_visible; ///< The number of visible units
    static inline constexpr size_t num_hidden  = desc::num_hidden;  ///< The number of hidden units

    static constexpr auto activation_function = desc::activation_function;                           ///< The layer's activation function
    static constexpr auto no_bias             = desc::parameters::template contains<dll::no_bias>(); ///< Disable the biases

    using w_initializer = typename desc::w_initializer; ///< The initializer for the weights
    using b_initializer = typename desc::b_initializer; ///< The initializer for the biases

    using input_one_t  = etl::fast_dyn_matrix<weight, num_visible>; ///< The type of one input
    using output_one_t = etl::fast_dyn_matrix<weight, num_hidden>;  ///< The type of one output
    using input_t      = std::vector<input_one_t>;                  ///< The type of the input
    using output_t     = std::vector<output_one_t>;                 ///< The type of the output

    using w_type = etl::fast_matrix<weight, num_visible, num_hidden>; ///< The type of the weights
    using b_type = etl::fast_matrix<weight, num_hidden>;              ///< The type of the biases

    //Weights and biases
    w_type w; ///< Weights
    b_type b; ///< Hidden biases

    //Backup Weights and biases
    std::unique_ptr<w_type> bak_w; ///< Backup Weights
    std::unique_ptr<b_type> bak_b; ///< Backup Hidden biases

    /*!
     * \brief Initialize a dense layer with basic weights.
     *
     * The weights are initialized from a normal distribution of
     * zero-mean and unit variance.
     */
    dense_layer_impl() : base_type() {
        w_initializer::initialize(w, input_size(), output_size());
        b_initializer::initialize(b, input_size(), output_size());
    }

    /*!
     * \brief Returns the input size of this layer
     */
    static constexpr size_t input_size() noexcept {
        return num_visible;
    }

    /*!
     * \brief Returns the output size of this layer
     */
    static constexpr size_t output_size() noexcept {
        return num_hidden;
    }

    /*!
     * \brief Returns the number of parameters of this layer
     */
    static constexpr size_t parameters() noexcept {
        // Weights + Biases
        return num_visible * num_hidden + num_hidden;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        if constexpr (activation_function == function::IDENTITY) {
            return "Dense";
        } else {
            char buffer[512];
            snprintf(buffer, 512, "Dense (%s)", to_string(activation_function).c_str());
            return {buffer};
        }
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        char buffer[512];

        if constexpr (activation_function == function::IDENTITY) {
            snprintf(buffer, 512, "Dense: %lu -> %lu", num_visible, num_hidden);
        } else {
            snprintf(buffer, 512, "Dense: %lu -> %s -> %lu", num_visible, to_string(activation_function).c_str(), num_hidden);
        }

        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {num_hidden};
    }

    /*!
     * \brief Apply the layer to the given batch of input.
     *
     * \param input A batch of input
     * \param output A batch of output that will be filled
     */
    template <typename H, typename V>
    void forward_batch(H&& output, const V& input) const {
        dll::auto_timer timer("dense:forward_batch");

        const auto Batch = etl::dim<0>(input);

        // Note: The compile-time Batch information is lost here, but it does
        // not matter for BLAS gemm computation

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        output = etl::reshape(input, Batch, num_visible) * w;

        if constexpr (!no_bias) {
            output = bias_add_2d(output, b);
        }

        output = f_activate<activation_function>(output);
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    output_one_t prepare_one_output() const {
        return {};
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    static output_t prepare_output(size_t samples) {
        return output_t{samples};
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
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
        dll::unsafe_auto_timer timer("dense:adapt_errors");

        if constexpr (activation_function != function::IDENTITY){
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
        dll::unsafe_auto_timer timer("dense:backward_batch");

        // The reshape has no overhead, so better than SFINAE for nothing
        constexpr auto Batch = etl::decay_traits<decltype(context.errors)>::template dim<0>();
        etl::reshape<Batch, num_visible>(output) = context.errors * etl::transpose(w);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        dll::unsafe_auto_timer timer("dense:compute_gradients");

        std::get<0>(context.up.context)->grad = batch_outer(context.input, context.errors);

        if constexpr (!no_bias) {
            std::get<1>(context.up.context)->grad = bias_batch_sum_2d(context.errors);
        }
    }
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<dense_layer_impl<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = true;  ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false;  ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief specialization of sgd_context for dense_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dense_layer_impl<Desc>, L> {
    using layer_t = dense_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr auto num_visible = layer_t::num_visible;
    static constexpr auto num_hidden  = layer_t::num_hidden;

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, num_visible> input;
    etl::fast_matrix<weight, batch_size, num_hidden> output;
    etl::fast_matrix<weight, batch_size, num_hidden> errors;

    sgd_context(const dense_layer_impl<Desc>& /* layer */)
            : output(0.0), errors(0.0) {}
};

} //end of dll namespace
