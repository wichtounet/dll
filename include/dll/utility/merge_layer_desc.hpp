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
 * \brief Describe a layer that merges layers together.
 */
template <size_t D_T, typename... Layers>
struct merge_layer_desc {
    static constexpr size_t D = D_T;

    /*!
     * \brief A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = merge_layer_impl<merge_layer_desc<D, Layers...>>;

    /*! The dynamic layer type */
    using dyn_layer_t = dyn_merge_layer_impl<dyn_merge_layer_desc<D, typename Layers::dyn_layer_t...>>;

    static_assert(sizeof...(Layers) > 0, "A merge layer must contain at least one layer");
};

/*!
 * \brief Describe a standard merge layer.
 */
template <size_t D, typename... Layers>
using merge_layer = typename merge_layer_desc<D, Layers...>::layer_t;

} //end of dll namespace
