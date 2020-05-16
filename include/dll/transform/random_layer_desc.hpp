//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Descriptor for randomization layer
 */
struct random_layer_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = random_layer_impl<random_layer_desc>;

    /*! The dynamic layer type */
    using dyn_layer_t = random_layer_impl<random_layer_desc>;
};

/*!
 * \brief Descriptor for randomization layer
 */
using random_layer = typename random_layer_desc::layer_t;

} //end of dll namespace
