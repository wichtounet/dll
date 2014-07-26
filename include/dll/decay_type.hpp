//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DECAY_TYPE_HPP
#define DLL_DECAY_TYPE_HPP

namespace dll {

/*!
 * \brief Define how weight decay is applied.
 */
enum class decay_type {
    NONE,           ///< No weight decay is applied during training
    L1,             ///< Apply L1 weight decay on weights
    L2,             ///< Apply L2 weight decay on weights
    L1_FULL,        ///< Apply L1 weight decay on weights and biases
    L2_FULL         ///< Apply L2 weight decay on weights and biases
};

} //end of dbn namespace

#endif