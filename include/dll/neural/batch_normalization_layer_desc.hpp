//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief A descriptor for a 2D Batch Normalization layer.
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
    using layer_t = batch_normalization_2d_layer<batch_normalization_layer_2d_desc<Input>>;

    /*!
     * The dynamic layer type
     */
    using dyn_layer_t = batch_normalization_2d_layer<batch_normalization_layer_2d_desc<Input>>;
};

/*!
 * \brief A descriptor for a 4D Batch Normalization layer.
 */
template<size_t K, size_t W, size_t H>
struct batch_normalization_layer_4d_desc {
    /*!
     * \brief Number of feature maps
     */
    static constexpr size_t Kernels = K;

    /*!
     * \brief Width of a feature map
     */
    static constexpr size_t Width = W;

    /*!
     * \brief Height of a feature map
     */
    static constexpr size_t Height = H;

    /*!
     * The layer type
     */
    using layer_t = batch_normalization_4d_layer<batch_normalization_layer_4d_desc<K, W, H>>;

    /*!
     * The dynamic layer type
     */
    using dyn_layer_t = batch_normalization_4d_layer<batch_normalization_layer_4d_desc<K, W, H>>;
};

} //end of dll namespace
