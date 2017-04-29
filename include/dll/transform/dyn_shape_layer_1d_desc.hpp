//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

struct dyn_shape_layer_1d_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = dyn_shape_layer_1d<dyn_shape_layer_1d_desc>;

    /*! The layer type */
    using dyn_layer_t = dyn_shape_layer_1d<dyn_shape_layer_1d_desc>;
};

} //end of dll namespace
