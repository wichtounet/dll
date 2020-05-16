//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Descriptor for a rectifier layer (abs)
 */
template <rectifier_method M = rectifier_method::ABS>
struct rectifier_layer_desc {
    static constexpr rectifier_method method = M; ///< The rectifier method to use

    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = rectifier_layer_impl<rectifier_layer_desc<M>>;

    /*! The dynamic layer type */
    using dyn_layer_t = rectifier_layer_impl<rectifier_layer_desc<M>>;
};

/*!
 * \brief Descriptor for a rectifier layer (abs)
 */
template <rectifier_method M = rectifier_method::ABS>
using rectifier_layer = typename rectifier_layer_desc<M>::layer_t;

} //end of dll namespace
