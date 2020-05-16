//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "pooling_layer_desc.hpp"

namespace dll {

/*!
 * \brief Description of a Dynamic Average Pooling two-dimensional layer.
 */
template <typename... Parameters>
struct dyn_avgp_2d_layer_desc : dyn_pooling_2d_layer_desc<Parameters...> {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The layer type */
    using layer_t = dyn_avgp_2d_layer_impl<dyn_avgp_2d_layer_desc<Parameters...>>;

    /*! The layer type */
    using dyn_layer_t = dyn_avgp_2d_layer_impl<dyn_avgp_2d_layer_desc<Parameters...>>;
};

/*!
 * \brief Description of a Dynamic Average Pooling three-dimensional layer.
 */
template <typename... Parameters>
struct dyn_avgp_3d_layer_desc : dyn_pooling_3d_layer_desc<Parameters...> {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The layer type */
    using layer_t = dyn_avgp_3d_layer_impl<dyn_avgp_3d_layer_desc<Parameters...>>;

    /*! The layer type */
    using dyn_layer_t = dyn_avgp_3d_layer_impl<dyn_avgp_3d_layer_desc<Parameters...>>;
};

/*!
 * \brief Description of a Dynamic Average Pooling two-dimensional layer.
 */
template <typename... Parameters>
using dyn_avgp_2d_layer = typename dyn_pooling_2d_layer_desc<Parameters...>::layer_t;

/*!
 * \brief Description of a Dynamic Average Pooling three-dimensional layer.
 */
template <typename... Parameters>
using dyn_avgp_3d_layer = typename dyn_pooling_3d_layer_desc<Parameters...>::layer_t;

} //end of dll namespace
