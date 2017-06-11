//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
struct dyn_upsample_layer_3d_desc : dyn_unpooling_layer_3d_desc<Parameters...> {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The RBM type */
    using layer_t = dyn_upsample_layer_3d<dyn_upsample_layer_3d_desc<Parameters...>>;

    /*! The RBM type */
    using dyn_layer_t = layer_t;
};

} //end of dll namespace
