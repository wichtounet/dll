//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Descriptor for layer that normalizes the input data to
 * zero-mean and unit-variance
 */
struct normalize_layer_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = normalize_layer_impl<normalize_layer_desc>;

    /*! The dynamic layer type */
    using dyn_layer_t = normalize_layer_impl<normalize_layer_desc>;
};

/*!
 * \brief Descriptor for layer that normalizes the input data to
 * zero-mean and unit-variance
 */
using normalize_layer = typename normalize_layer_desc::layer_t;

} //end of dll namespace
