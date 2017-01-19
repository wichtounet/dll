//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

struct dyn_lcn_layer_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = dyn_lcn_layer<dyn_lcn_layer_desc>;

    /*! The layer type */
    using dyn_layer_t = dyn_lcn_layer<dyn_lcn_layer_desc>;
};

} //end of dll namespace
