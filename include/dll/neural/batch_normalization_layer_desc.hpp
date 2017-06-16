//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief A descriptor for a dropout layer.
 */
template<size_t I>
struct batch_normalization_layer_2d_desc {
    /*!
     * \brief Input Size
     */
    static constexpr size_t Input  = I;

    /*!
     * The layer type
     */
    using layer_t = batch_normalization_2d_layer<batch_normalization_2d_layer_desc<Input>>;

    /*!
     * The dynamic layer type
     */
    using dyn_layer_t = batch_normalization_2d_layer<batch_normalization_2d_layer_desc<Input>>;
};

} //end of dll namespace
