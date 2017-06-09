//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_conf.hpp"
#include "dll/util/tmp.hpp"

namespace dll {

/*!
 * \brief Describe a standard convolutional layer with 'same' padding.
 */
template <size_t NC_T, size_t NV_1, size_t NV_2, size_t K_T, size_t NW_1, size_t NW_2, typename... Parameters>
struct conv_same_desc {
    static constexpr size_t NV1 = NV_1; ///< The first dimension of the input
    static constexpr size_t NV2 = NV_2; ///< The second dimension of the input
    static constexpr size_t NW1 = NW_1; ///< The first dimension of the output
    static constexpr size_t NW2 = NW_2; ///< The second dimension of the output
    static constexpr size_t NC  = NC_T; ///< The number of input channels
    static constexpr size_t K   = K_T;  ///< The number of filters

    /*!
     * \brief A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    static constexpr auto activation_function = detail::get_value<activation<function::SIGMOID>, Parameters...>::value; ///< The layer's activation function
    static constexpr auto w_initializer       = detail::get_value<initializer<initializer_type::LECUN>, Parameters...>::value; ///< The initializer for the weights
    static constexpr auto b_initializer       = detail::get_value<initializer_bias<initializer_type::LECUN>, Parameters...>::value; ///< The initializer for the biases

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*! The conv type */
    using layer_t = conv_same_layer<conv_same_desc<NC_T, NV_1, NV_2, K_T, NW_1, NW_2, Parameters...>>;

    /*! The conv type */
    using dyn_layer_t = dyn_conv_same_layer<dyn_conv_same_desc<Parameters...>>;

    static_assert(NV1 > 0, "A matrix of at least 1x1 is necessary for the visible units");
    static_assert(NV2 > 0, "A matrix of at least 1x1 is necessary for the visible units");
    static_assert(NW1 > 0, "A matrix of at least 1x1 is necessary for the weights");
    static_assert(NW2 > 0, "A matrix of at least 1x1 is necessary for the weights");
    static_assert(NC > 0, "At least one channel is necessary");
    static_assert(K > 0, "At least one group is necessary");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id, activation_id, initializer_id, initializer_bias_id>, Parameters...>::value,
        "Invalid parameters type for conv_same_desc");
};

} //end of dll namespace
