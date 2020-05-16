//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Descriptor for a dynamic Local Contrast Normalization (LCN) layer
 */
struct dyn_lcn_layer_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = dyn_lcn_layer_impl<dyn_lcn_layer_desc>;

    /*! The layer type */
    using dyn_layer_t = dyn_lcn_layer_impl<dyn_lcn_layer_desc>;
};

/*!
 * \brief Descriptor for a dynamic Local Contrast Normalization (LCN) layer
 */
using dyn_lcn_layer = typename dyn_lcn_layer_desc::layer_t;

} //end of dll namespace
