//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_BIAS_MODE_HPP
#define DLL_BIAS_MODE_HPP

namespace dll {

/*!
 * \brief Several modes for biases computation in convolutional RBM
 */
enum class bias_mode {
    NONE,
    SIMPLE
};

} //end of dll namespace

#endif