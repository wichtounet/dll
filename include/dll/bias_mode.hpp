//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Several modes for biases computation in convolutional RBM
 */
enum class bias_mode {
    NONE,  ///< The sparsity bias is not computed
    SIMPLE ///< The sparsity bias is computed on the visible biases
};

} //end of dll namespace
