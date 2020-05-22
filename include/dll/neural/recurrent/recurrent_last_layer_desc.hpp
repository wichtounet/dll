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
 * \brief Descriptor for a recurrent last layer
 */
template <size_t TS_T, size_t HU_T, typename... Parameters>
struct recurrent_last_layer_desc {
    static constexpr size_t time_steps      = TS_T; ///< The number of time steps
    static constexpr size_t hidden_units    = HU_T; ///< The number of hidden units

    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*! The fast type */
    using layer_t = recurrent_last_layer_impl<recurrent_last_layer_desc<TS_T, HU_T, Parameters...>>;

    /*! The dynamic type */
    using dyn_layer_t = dyn_recurrent_last_layer_impl<dyn_recurrent_last_layer_desc<Parameters...>>;

    static_assert(time_steps > 0, "There must be at least 1 time step");
    static_assert(hidden_units > 0, "There must be at least 1 hidden unit");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<
            weight_type_id>,
            Parameters...>,
        "Invalid parameters type for recurrent_last_layer_desc");
};

/*!
 * \brief Describe a dense layer
 */
template <size_t TS_T, size_t HU_T, typename... Parameters>
using recurrent_last_layer = typename recurrent_last_layer_desc<TS_T, HU_T, Parameters...>::layer_t;

} //end of dll namespace
