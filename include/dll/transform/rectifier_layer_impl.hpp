//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "transform_layer.hpp"

namespace dll {

/*!
 * \brief Configuraable rectifier layer.
 *
 * Use abs as a rectifier by default
 */
template <typename Desc>
struct rectifier_layer_impl : transform_layer<rectifier_layer_impl<Desc>> {
    using desc        = Desc;                                        ///< The descriptor type
    using base_type   = transform_layer<rectifier_layer_impl<Desc>>; ///< The base type
    using this_type   = rectifier_layer_impl<Desc>;                  ///< The type of this layer
    using layer_t     = this_type;                                   ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;                  ///< The dynamic version of this layer

    static constexpr rectifier_method method = desc::method; ///< The rectifier method

    static_assert(method == rectifier_method::ABS, "Only ABS rectifier has been implemented");

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        return "Rectifier";
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        return "Rectifier";
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void forward_batch(Output& output, const Input& input) {
        if (method == rectifier_method::ABS) {
            output = etl::abs(input);
        }
    }
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const rectifier_method rectifier_layer_impl<Desc>::method;

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<rectifier_layer_impl<Desc>> {
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
 * \brief Specialization of sgd_context for rectifier_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, rectifier_layer_impl<Desc>, L> {
    using layer_t          = rectifier_layer_impl<Desc>;                            ///< The current layer type
    using previous_layer   = typename DBN::template layer_type<L - 1>;          ///< The previous layer type
    using previous_context = sgd_context<DBN, previous_layer, L - 1>;           ///< The previous layer's context
    using inputs_t         = decltype(std::declval<previous_context>().output); ///< The type of inputs

    inputs_t input;  ///< A batch of input
    inputs_t output; ///< A batch of output
    inputs_t errors; ///< A batch of errors

    sgd_context(const layer_t& /*layer*/){}
};

} //end of dll namespace
