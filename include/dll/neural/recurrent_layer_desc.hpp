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
 * \brief Descriptor for a recurrent layer
 */
template <size_t TS_T, size_t SL_T, size_t HU_T, typename... Parameters>
struct recurrent_layer_desc {
    static constexpr size_t time_steps      = TS_T; ///< The number of time steps
    static constexpr size_t sequence_length = SL_T; ///< The length of the sequences
    static constexpr size_t hidden_units    = HU_T; ///< The number of hidden units

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

    using w_initializer = detail::get_type_t<initializer<init_lecun>, Parameters...>;     ///< The initializer for the weights

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*! The dense type */
    using layer_t = recurrent_layer_impl<recurrent_layer_desc<TS_T, SL_T, HU_T, Parameters...>>;

    /*! The dense type */
    using dyn_layer_t = recurrent_layer_impl<recurrent_layer_desc<TS_T, SL_T, HU_T, Parameters...>>;
    //TODO Fix this

    static_assert(time_steps > 0, "There must be at least 1 time step");
    static_assert(sequence_length > 0, "The sequence must be at least 1 element");
    static_assert(hidden_units > 0, "There must be at least 1 hidden unit");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<
            weight_type_id, activation_id, initializer_id, truncate_id>,
            Parameters...>,
        "Invalid parameters type for recurrent_layer_desc");
};

/*!
 * \brief Describe a dense layer
 */
template <size_t TS_T, size_t SL_T, size_t HU_T, typename... Parameters>
using recurrent_layer = typename recurrent_layer_desc<TS_T, SL_T, HU_T, Parameters...>::layer_t;

} //end of dll namespace
