//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Descriptor for a scaling layer (multiply by A/B)
 */
template <int A_T, int B_T>
struct scale_layer_desc {
    static constexpr int A = A_T; ///< The scaling multiplicative factor
    static constexpr int B = B_T; ///< The scaling dividive factor

    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = scale_layer_impl<scale_layer_desc<A, B>>;

    /*! The dynamic layer type */
    using dyn_layer_t = scale_layer_impl<scale_layer_desc<A, B>>;
};

/*!
 * \brief Descriptor for a scaling layer (multiply by A/B)
 */
template <int A_T, int B_T>
using scale_layer = typename scale_layer_desc<A_T, B_T>::layer_t;

} //end of dll namespace
