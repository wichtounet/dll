//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "unpooling_layer_desc.hpp"

namespace dll {

template <size_t T_I1, size_t T_I2, size_t T_I3, size_t T_C1, size_t T_C2, size_t T_C3, typename... Parameters>
struct upsample_layer_3d_desc : unpooling_layer_3d_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_C3, Parameters...> {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The layer type */
    using layer_t = upsample_layer_3d<upsample_layer_3d_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_C3, Parameters...>>;

    /*! The RBM type */
    using dyn_layer_t = dyn_upsample_layer_3d<dyn_upsample_layer_3d_desc<Parameters...>>;
};

} //end of dll namespace
