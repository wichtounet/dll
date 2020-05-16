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
 * \brief Describe a layer that groups layers together.
 */
template <typename... Layers>
struct group_layer_desc {
    /*!
     * \brief A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = group_layer_impl<group_layer_desc<Layers...>>;

    /*! The dynamic layer type */
    using dyn_layer_t = dyn_group_layer_impl<dyn_group_layer_desc<typename Layers::dyn_layer_t...>>;

    static_assert(sizeof...(Layers) > 0, "A group layer must contain at least one layer");
};

/*!
 * \brief Describe a standard group layer.
 */
template <typename... Layers>
using group_layer = typename group_layer_desc<Layers...>::layer_t;

} //end of dll namespace
