//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"  // The traits
#include "dll/neural_layer.hpp" // The base class
#include "dll/util/timers.hpp"  // For auto_timer

namespace dll {

/*!
 * \brief Standard dense layer of neural network.
 */
template <typename Desc>
struct dyn_dense_layer_impl final : neural_layer<dyn_dense_layer_impl<Desc>, Desc> {
    using desc        = Desc;                          ///< The descriptor of the layer
    using weight      = typename desc::weight;         ///< The data type for this layer
    using this_type   = dyn_dense_layer_impl<desc>;    ///< The type of this layer
    using base_type   = neural_layer<this_type, desc>; ///< The type of the base type
    using layer_t     = this_type;                     ///< The type of this layer
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic type of this layer

    static constexpr auto activation_function = desc::activation_function;                           ///< The layer's activation function
    static constexpr auto no_bias             = desc::parameters::template contains<dll::no_bias>(); ///< Disable the biases

    using w_initializer = typename desc::w_initializer; ///< The initializer for the weights
    using b_initializer = typename desc::b_initializer; ///< The initializer for the biases

    using input_one_t  = etl::dyn_matrix<weight, 1>; ///< The type of one input
    using output_one_t = etl::dyn_matrix<weight, 1>; ///< The type of one output
    using input_t      = std::vector<input_one_t>;   ///< The type of the input
    using output_t     = std::vector<output_one_t>;  ///< The type of the output

    using w_type = etl::dyn_matrix<weight, 2>; ///< The type of the weights
    using b_type = etl::dyn_matrix<weight, 1>; ///< The type of the biases

    //Weights and biases
    w_type w; ///< Weights
    b_type b; ///< Hidden biases

    //Backup Weights and biases
    std::unique_ptr<w_type> bak_w; ///< Backup Weights
    std::unique_ptr<b_type> bak_b; ///< Backup Hidden biases

    size_t num_visible; ///< The number of visible units
    size_t num_hidden;  ///< The number of hidden units

    dyn_dense_layer_impl() : base_type() {}

    /*!
     * \brief Initialize the dynamic layer
     */
    void init_layer(size_t nv, size_t nh) {
        num_visible = nv;
        num_hidden  = nh;

        w = etl::dyn_matrix<weight, 2>(num_visible, num_hidden);
        b = etl::dyn_matrix<weight, 1>(num_hidden);

        w_initializer::initialize(w, input_size(), output_size());
        b_initializer::initialize(b, input_size(), output_size());
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
        // Weights + Biases
        return num_visible * num_hidden + num_hidden;
    }

    /*!
     * \brief Returns a full description of the layer
     * \return an std::string containing a full description of the layer
     */
    std::string to_short_string([[maybe_unused]] std::string pre = "") const {
        if constexpr (activation_function == function::IDENTITY) {
            return "Dense (dyn)";
        } else {
            char buffer[512];
            snprintf(buffer, 512, "Dense(%s) (dyn)", to_string(activation_function).c_str());
            return {buffer};
        }
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_full_string([[maybe_unused]] std::string pre = "") const {
        char buffer[512];

        if constexpr (activation_function == function::IDENTITY) {
            snprintf(buffer, 512, "Dense(dyn): %lu -> %lu", num_visible, num_hidden);
        } else {
            snprintf(buffer, 512, "Dense(dyn): %lu -> %s -> %lu", num_visible, to_string(activation_function).c_str(), num_hidden);
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
        dll::auto_timer timer("dense:forward");

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
        dll::unsafe_auto_timer timer("dense:errors");

        context.errors = f_derivative<activation_function>(context.output) >> context.errors;
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        dll::unsafe_auto_timer timer("dense:backward");

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
        dll::unsafe_auto_timer timer("dense:gradients");

        std::get<0>(context.up.context)->grad = batch_outer(context.input, context.errors);

        if constexpr (!no_bias) {
            std::get<1>(context.up.context)->grad = bias_batch_sum_2d(context.errors);
        }
    }
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<dyn_dense_layer_impl<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = true;  ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = true;  ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dyn_dense_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_dense_layer_impl<Desc>, L> {
    using layer_t = dyn_dense_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::dyn_matrix<weight, 2> input;
    etl::dyn_matrix<weight, 2> output;
    etl::dyn_matrix<weight, 2> errors;

    sgd_context(const layer_t& layer) : input(batch_size, layer.num_visible, 0.0), output(batch_size, layer.num_hidden, 0.0), errors(batch_size, layer.num_hidden, 0.0) {}
};


} //end of dll namespace
