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
 * \brief Activation layer
 *
 * Applies an activation function to the output of one layer.
 */
template <typename Desc>
struct activation_layer_impl : transform_layer<activation_layer_impl<Desc>> {
    using desc        = Desc;                                         ///< The descriptor type
    using base_type   = transform_layer<activation_layer_impl<Desc>>; ///< The base type
    using this_type   = activation_layer_impl<Desc>;                  ///< This layer's type
    using layer_t     = this_type;                                    ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;                   ///< The dynamic version of this layer

    static constexpr function activation_function = desc::activation_function;

    activation_layer_impl() = default;

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        char buffer[128];
        snprintf(buffer, 128, "Activation(%s)", to_string(activation_function).c_str());
        return {buffer};
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        char buffer[128];
        snprintf(buffer, 128, "Activation(%s)", to_string(activation_function).c_str());
        return {buffer};
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void forward_batch(Output& output, const Input& input) {
        output = f_activate<activation_function>(input);
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
        if constexpr (activation_function != function::IDENTITY) {
            output = f_derivative<activation_function>(context.output) >> context.errors;
        }
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
struct layer_base_traits<activation_layer_impl<Desc>> {
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
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for activation_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, activation_layer_impl<Desc>, L> {
    using layer_t          = activation_layer_impl<Desc>;                            ///< The current layer type
    using previous_layer   = typename DBN::template layer_type<L - 1>;          ///< The previous layer type
    using previous_context = sgd_context<DBN, previous_layer, L - 1>;           ///< The previous layer's context
    using inputs_t         = decltype(std::declval<previous_context>().output); ///< The type of inputs

    inputs_t input;  ///< A batch of input
    inputs_t output; ///< A batch of output
    inputs_t errors; ///< A batch of errors

    sgd_context(const layer_t& /*layer*/){}
};

} //end of dll namespace
