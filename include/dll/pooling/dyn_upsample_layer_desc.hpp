//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "unpooling_layer_desc.hpp"

namespace dll {

/*!
 * \brief Descriptor for a dynamic 3D upsample layer
 */
template <typename... Parameters>
struct dyn_upsample_3d_layer_desc : dyn_unpooling_3d_layer_desc<Parameters...> {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The RBM type */
    using layer_t = dyn_upsample_3d_layer_impl<dyn_upsample_3d_layer_desc<Parameters...>>;

    /*! The RBM type */
    using dyn_layer_t = layer_t;
};

/*!
 * \brief Descriptor for a dynamic 3D upsample layer
 */
template <typename... Parameters>
using dyn_upsample_3d_layer = typename dyn_upsample_3d_layer_desc<Parameters...>::layer_t;

} //end of dll namespace
