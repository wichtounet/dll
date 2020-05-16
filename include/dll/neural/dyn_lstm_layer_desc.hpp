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
 * \brief Descriptor for a LSTM recurrent layer
 */
template <typename... Parameters>
struct dyn_lstm_layer_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*!
     * \brief The activation function
     */
    static constexpr auto activation_function = detail::get_value_v<activation<function::TANH>, Parameters...>;            ///< The layer's activation function

    /*!
     * \brief The BPTT steps
     */
    static constexpr size_t Truncate    = detail::get_value_v<truncate<0>, Parameters...>;

    using w_initializer  = detail::get_type_t<rnn_initializer_w<init_lecun>, Parameters...>;     ///< The initializer for the W weights
    using u_initializer  = detail::get_type_t<rnn_initializer_u<init_lecun>, Parameters...>;     ///< The initializer for the U weights
    using b_initializer  = detail::get_type_t<initializer_bias<init_zero>, Parameters...>;       ///< The initializer for the biases
    using fb_initializer = detail::get_type_t<initializer_forget_bias<init_one>, Parameters...>; ///< The initializer for the forget biases

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*! The dense type */
    using layer_t = dyn_lstm_layer_impl<dyn_lstm_layer_desc<Parameters...>>;

    /*! The dense type */
    using dyn_layer_t = dyn_lstm_layer_impl<dyn_lstm_layer_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<
            weight_type_id, activation_id, rnn_initializer_w_id, rnn_initializer_u_id,
            initializer_bias_id, initializer_forget_bias_id, truncate_id, last_only_id>,
            Parameters...>,
        "Invalid parameters type for dyn_lstm_layer_desc");
};

/*!
 * \brief Describe a dynamic LSTM layer
 */
template <typename... Parameters>
using dyn_lstm_layer = typename dyn_lstm_layer_desc<Parameters...>::layer_t;

} //end of dll namespace
