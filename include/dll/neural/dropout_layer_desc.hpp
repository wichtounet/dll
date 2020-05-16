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
template <size_t D>
struct dropout_layer_desc {
    static_assert(D > 0, "Invalid dropout factor");
    static_assert(D < 101, "Invalid dropout factor");

    /*!
     * \brief Drop percentage
     */
    static constexpr size_t Drop  = D;

    /*!
     * The layer type
     */
    using layer_t = dropout_layer_impl<dropout_layer_desc<D>>;

    /*!
     * The dynamic layer type
     */
    using dyn_layer_t = dyn_dropout_layer_impl<dyn_dropout_layer_desc>;
};

/*!
 * \brief A descriptor for a dropout layer.
 */
template <size_t D>
using dropout_layer = typename dropout_layer_desc<D>::layer_t;

} //end of dll namespace
