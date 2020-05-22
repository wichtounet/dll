//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief A descriptor for an activation layer.
 *
 * Such a layer only applies an activation function to its inputs
 * and has no weights.
 */
template <dll::function F = dll::function::SIGMOID>
struct activation_layer_desc {
    /*!
     * \brief The layer's activation function
     */
    static constexpr function activation_function = F;

    /*!
     * The layer type
     */
    using layer_t = activation_layer_impl<activation_layer_desc<F>>;

    /*!
     * The dynamic layer type
     */
    using dyn_layer_t = activation_layer_impl<activation_layer_desc<F>>;
};

/*!
 * \brief A descriptor for an activation layer.
 *
 * Such a layer only applies an activation function to its inputs
 * and has no weights.
 */
template <dll::function F = dll::function::SIGMOID>
using activation_layer = typename activation_layer_desc<F>::layer_t;

} //end of dll namespace
