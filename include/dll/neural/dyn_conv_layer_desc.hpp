//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_conf.hpp"
#include "dll/util/tmp.hpp"

namespace dll {

/*!
 * \brief Describe a standard dynamic convolutional layer.
 */
template <typename... Parameters>
struct dyn_conv_layer_desc {
    /*!
     * \brief A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    static constexpr auto activation_function = detail::get_value_v<activation<function::SIGMOID>, Parameters...>;            ///< The layer's activation function

    using w_initializer = detail::get_type_t<initializer<init_lecun>, Parameters...>;     ///< The initializer for the weights
    using b_initializer = detail::get_type_t<initializer_bias<init_zero>, Parameters...>; ///< The initializer for the biases

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*! The conv type */
    using layer_t = dyn_conv_layer_impl<dyn_conv_layer_desc<Parameters...>>;

    /*! The conv type */
    using dyn_layer_t = dyn_conv_layer_impl<dyn_conv_layer_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<weight_type_id, activation_id, initializer_id, initializer_bias_id, no_bias_id>, Parameters...>,
        "Invalid parameters type for dyn_conv_layer_desc");
};

/*!
 * \brief Describe a standard dynamic convolutional layer.
 */
template <typename... Parameters>
using dyn_conv_layer = typename dyn_conv_layer_desc<Parameters...>::layer_t;

} //end of dll namespace
