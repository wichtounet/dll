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
template <typename... Parameters>
struct dyn_recurrent_last_layer_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*! The fast type */
    using layer_t = dyn_recurrent_last_layer_impl<dyn_recurrent_last_layer_desc<Parameters...>>;

    /*! The dynamic type */
    using dyn_layer_t = dyn_recurrent_last_layer_impl<dyn_recurrent_last_layer_desc<Parameters...>>;

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
template <typename... Parameters>
using dyn_recurrent_last_layer = typename dyn_recurrent_last_layer_desc<Parameters...>::layer_t;

} //end of dll namespace
