//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief A descriptor for a dropout layer.
 */
struct dyn_dropout_layer_desc {
    /*!
     * The layer type
     */
    using layer_t = dyn_dropout_layer_impl<dyn_dropout_layer_desc>;

    /*!
     * The dynamic layer type
     */
    using dyn_layer_t = dyn_dropout_layer_impl<dyn_dropout_layer_desc>;
};

/*!
 * \brief A descriptor for a dropout layer.
 */
using dyn_dropout_layer = typename dyn_dropout_layer_desc::layer_t;

} //end of dll namespace
