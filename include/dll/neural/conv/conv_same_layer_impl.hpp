//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/neural_layer.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Standard convolutional layer of neural network.
 */
template <typename Desc>
struct conv_same_layer_impl final : neural_layer<conv_same_layer_impl<Desc>, Desc> {
    using desc      = Desc; ///< The descriptor of the layer
    using weight    = typename desc::weight; ///< The data type for this layer
    using this_type = conv_same_layer_impl<desc>; ///< The type of this layer
    using base_type = neural_layer<this_type, desc>;
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    static inline constexpr size_t NV1 = desc::NV1; ///< The first dimension of the visible units
    static inline constexpr size_t NV2 = desc::NV2; ///< The second dimension of the visible units
    static inline constexpr size_t NW1 = desc::NW1; ///< The first dimension of the filter
    static inline constexpr size_t NW2 = desc::NW2; ///< The second dimension of the filter
    static inline constexpr size_t NC  = desc::NC;  ///< The number of input channels
    static inline constexpr size_t K   = desc::K;   ///< The number of filters

    static inline constexpr size_t NH1 = NV1; //By definition
    static inline constexpr size_t NH2 = NV2; //By definition

    static inline constexpr size_t P1 = (NW1 - 1) / 2;
    static inline constexpr size_t P2 = (NW2 - 1) / 2;

    static constexpr auto activation_function = desc::activation_function; ///< The layer's activation function

    using w_initializer = typename desc::w_initializer; ///< The initializer for the weights
    using b_initializer = typename desc::b_initializer; ///< The initializer for the biases

    static_assert(NW1 % 2 == 1, "conv_same_layer_impl only works with odd-sized filters");
    static_assert(NW2 % 2 == 1, "conv_same_layer_impl only works with odd-sized filters");

    using input_one_t  = etl::fast_dyn_matrix<weight, NC, NV1, NV2>; ///< The type of one input
    using output_one_t = etl::fast_dyn_matrix<weight, K, NH1, NH2>; ///< The type of one output
    using input_t      = std::vector<input_one_t>; ///< The type of the input
    using output_t     = std::vector<output_one_t>; ///< The type of the output

    using w_type = etl::fast_matrix<weight, K, NC, NW1, NW2>; ///< The type of the weights
    using b_type = etl::fast_matrix<weight, K>; ///< The type of the biases

    //Weights and biases
    w_type w; ///< Weights
    b_type b; ///< Hidden biases

    //Backup weights and biases
    std::unique_ptr<w_type> bak_w; ///< Backup Weights
    std::unique_ptr<b_type> bak_b; ///< Backup Hidden biases

    /*!
     * \brief Initialize a conv layer with basic weights.
     */
    conv_same_layer_impl() : base_type() {
        w_initializer::initialize(w, input_size(), output_size());
        b_initializer::initialize(b, input_size(), output_size());
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    static constexpr size_t input_size() noexcept {
        return NC * NV1 * NV2;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    static constexpr size_t output_size() noexcept {
        return K * NH1 * NH2;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    static constexpr size_t parameters() noexcept {
        return K * NW1 * NW2;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        char buffer[1024];
        snprintf(buffer, 1024, "Conv(same)(%s)", to_string(activation_function).c_str());
        return {buffer};
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        char buffer[1024];
        snprintf(buffer, 1024, "Conv(same): %lux%lux%lu -> (%lux%lux%lu) -> %s -> %lux%lux%lu", NC, NV1, NV2, K, NW1, NW2, to_string(activation_function).c_str(), K, NH1, NH2);
        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {K, NH1, NH2};
    }

    /*!
     * \brief Apply the layer to the given batch of input.
     *
     * \param input A batch of input
     * \param output A batch of output that will be filled
     */
    template <typename H1, typename V>
    void forward_batch(H1&& output, const V& v) const {
        dll::auto_timer timer("conv:forward_batch");

        if constexpr (etl::dimensions<V>() == 4) {
            output = etl::ml::convolution_forward<1, 1, P1, P2>(v, w);
        } else {
            output = etl::ml::convolution_forward<1, 1, P1, P2>(etl::reshape(v, etl::dim<0>(v), NC, NV1, NV2), w);
        }

        output = bias_add_4d(output, b);
        output = f_activate<activation_function>(output);
    }

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
    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(NC, NV1, NV2, K, NW1, NW2);
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
        dll::auto_timer timer("conv_same:adapt_errors");

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
        dll::auto_timer timer("conv_same:backward_batch");

        output = etl::ml::convolution_backward<1, 1, P1, P2>(context.errors, w);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        dll::auto_timer timer("conv_same:compute_gradients");

        std::get<0>(context.up.context)->grad = etl::ml::convolution_backward_filter<1, 1, P1, P2>(context.input, context.errors);
        std::get<1>(context.up.context)->grad = etl::bias_batch_sum_4d(context.errors);
    }
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<conv_same_layer_impl<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false;  ///< Indicates if the layer is dense
    static constexpr bool is_conv       = true; ///< Indicates if the layer is convolutional
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
 * \brief Specialization of the sgd_context for conv_same_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, conv_same_layer_impl<Desc>, L> {
    using layer_t = conv_same_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr size_t NV1 = layer_t::NV1;
    static constexpr size_t NV2 = layer_t::NV2;
    static constexpr size_t NC  = layer_t::NC;
    static constexpr size_t K   = layer_t::K;

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, NC, NV1, NV2> input;
    etl::fast_matrix<weight, batch_size, K,  NV1, NV2> output;
    etl::fast_matrix<weight, batch_size, K,  NV1, NV2> errors;

    sgd_context(const layer_t& /* layer */)
            : output(0.0), errors(0.0) {}
};

} //end of dll namespace
