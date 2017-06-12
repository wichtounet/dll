//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief The updater type for gradient descent
 */
enum class updater_type {
    SGD,     ///< The basic updater for SGD
    MOMENTUM ///< Use momentum for SGD
};

} //end of dll namespace
