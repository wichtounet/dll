//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_LR_DRIVER_TYPE_HPP
#define DLL_LR_DRIVER_TYPE_HPP

namespace dll {

/*!
 * \brief Define how the learning rate is adapted
 */
enum class lr_driver_type {
    FIXED, ///< The learning rate never changes
    BOLD,  ///< Bold driver of the learning rate
    STEP   ///< Step drive of the learning rate
};

} //end of dll namespace

#endif
