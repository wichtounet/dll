//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "pooling_layer_desc.hpp"

namespace dll {

template <typename... Parameters>
struct dyn_avgp_layer_2d_desc : dyn_pooling_layer_2d_desc<Parameters...> {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The layer type */
    using layer_t = dyn_avgp_layer_2d<dyn_avgp_layer_2d_desc<Parameters...>>;

    /*! The layer type */
    using dyn_layer_t = dyn_avgp_layer_2d<dyn_avgp_layer_2d_desc<Parameters...>>;
};

template <typename... Parameters>
struct dyn_avgp_layer_3d_desc : dyn_pooling_layer_3d_desc<Parameters...> {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The layer type */
    using layer_t = dyn_avgp_layer_3d<dyn_avgp_layer_3d_desc<Parameters...>>;

    /*! The layer type */
    using dyn_layer_t = dyn_avgp_layer_3d<dyn_avgp_layer_3d_desc<Parameters...>>;
};

} //end of dll namespace
