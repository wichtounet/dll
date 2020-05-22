//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/transform/transform_layer.hpp"

namespace dll {

/*!
 * \brief Dropout layer
 */
template <typename Desc>
struct dyn_dropout_layer_impl : transform_layer<dyn_dropout_layer_impl<Desc>> {
    using desc        = Desc;                         ///< The descriptor type
    using this_type   = dyn_dropout_layer_impl<Desc>; ///< This layer's type
    using base_type   = transform_layer<this_type>;   ///< The base type
    using layer_t     = this_type;                    ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;   ///< The dynamic version of this layer

    float p; ///< The dropout probability

    using dropout_t = decltype(etl::state_inverted_dropout_mask(dll::rand_engine(), p));

    dropout_t* dropout = nullptr; ///< The dropout mask generator (ETL)

    dyn_dropout_layer_impl() = default;

    /*!
     * \brief Delete the layer and frees all allocated resources
     */
    ~dyn_dropout_layer_impl() {
        if (dropout) {
            delete dropout;
        }
    }

    /*!
     * \brief Initialize the dynamic layer
     */
    void init_layer(float p) {
        this->p = p;

        dropout = new dropout_t(dll::rand_engine(), p);
    }

    /*!
     * \brief Returns a full string representation of the layer
     */
    std::string to_short_string([[maybe_unused]] std::string pre = "") const {
        char buffer[128];
        snprintf(buffer, 128, "Dropout(%.2f)(dyn)", p);
        return {buffer};
    }

    /*!
     * \brief Returns a full string representation of the layer
     */
    std::string to_full_string([[maybe_unused]] std::string pre = "") const {
        char buffer[128];
        snprintf(buffer, 128, "Dropout(%.2f)(dyn)", p);
        return {buffer};
    }

    using base_type::test_forward_batch;

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void forward_batch(Output& output, const Input& input) {
        output = input;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void test_forward_batch(Output& output, const Input& input) {
        dll::auto_timer timer("dropout:test:forward");

        output = input;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void train_forward_batch(Output& output, const Input& input) const {
        dll::auto_timer timer("dropout:train:forward");

        // For performance reasoons, we do on two pass since dropout is not
        // threadsafe and and not vectorizable
        output = *dropout;
        output = output >> input;
    }

    /*!
     * \brief Adapt the errors, called before backpropagation of the errors.
     *
     * This must be used by layers that have both an activation fnction and a non-linearity.
     *
     * \param context the training context
     */
    template <typename C>
    void adapt_errors([[maybe_unused]] C& context) const {}

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        dll::unsafe_auto_timer timer("dropout:backward");

        output = context.errors;
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template <typename C>
    void compute_gradients([[maybe_unused]] C& context) const {}
};

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<dyn_dropout_layer_impl<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = true;  ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = true;  ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dyn_dropout_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_dropout_layer_impl<Desc>, L> {
    using layer_t          = dyn_dropout_layer_impl<Desc>;                      ///< The current layer type
    using previous_layer   = typename DBN::template layer_type<L - 1>;          ///< The previous layer type
    using previous_context = sgd_context<DBN, previous_layer, L - 1>;           ///< The previous layer's context
    using inputs_t         = decltype(std::declval<previous_context>().output); ///< The type of inputs

    inputs_t input;  ///< A batch of input
    inputs_t output; ///< A batch of output
    inputs_t errors; ///< A batch of errors

    sgd_context(const layer_t& /*layer*/){}
};

} //end of dll namespace
