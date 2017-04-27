//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <size_t C_T, size_t H_T, size_t W_T>
struct shape_layer_3d_desc {
    static constexpr const size_t C = C_T; ///< The size
    static constexpr const size_t H = H_T; ///< The size
    static constexpr const size_t W = W_T; ///< The size

    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*!
     * The layer type
     */
    using layer_t = shape_layer_3d<shape_layer_3d_desc<C_T, H_T, W_T>>;

    /*!
     * The dynamic layer type
     */
    //TODO using dyn_layer_t = binarize_layer<binarize_layer_desc<T_T>>;
};

} //end of dll namespace
