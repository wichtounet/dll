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
 * \brief Describe a standard embedding layer.
 */
template <typename... Parameters>
struct dyn_embedding_layer_desc {
    /*!
     * \brief A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    using w_initializer = detail::get_type_t<initializer<init_uniform<constant(-1.0), constant(1.0)>>, Parameters...>;     ///< The initializer for the weights

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*! The embedding type */
    using layer_t = dyn_embedding_layer_impl<dyn_embedding_layer_desc<Parameters...>>;

    /*! The embedding type */
    using dyn_layer_t = dyn_embedding_layer_impl<dyn_embedding_layer_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<weight_type_id, initializer_id>, Parameters...>,
        "Invalid parameters type for dyn_embedding_layer_desc");
};

/*!
 * \brief Describe a standard embedding layer.
 */
template <typename... Parameters>
using dyn_embedding_layer = typename dyn_embedding_layer_desc<Parameters...>::layer_t;

} //end of dll namespace
