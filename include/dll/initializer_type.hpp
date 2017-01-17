//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \brief Initialization enumeration
 */

#pragma once

namespace dll {

/*!
 * \brief Define how weight decay is applied.
 */
enum class initializer_type {
    NONE,       ///< Not initialized (undefined values)
    ZERO,       ///< All initialized to zero
    GAUSSIAN,   ///< Initialization to N(0, 0.01)
    UNIFORM,    ///< Initialization to U(-0.05, 0.05)
    LECUN,      ///< Initialization to N(0, 1/sqrt(Nin))
    XAVIER,     ///< Initialization to N(0, 1/Nin)
    XAVIER_FULL ///< Initialization to N(0, 2/(Nin + Nout))
};

} //end of dll namespace
