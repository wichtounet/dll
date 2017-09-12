//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
    NONE,           ///< Not initialized (undefined values)
    ZERO,           ///< All initialized to 0.0
    ONE,            ///< All initialized to 1.0
    CONSTANT_01,    ///< All initialized to 0.1
    CONSTANT_001,   ///< All initialized to 0.01
    SMALL_GAUSSIAN, ///< Initialization to N(0, 0.01)
    GAUSSIAN,       ///< Initialization to N(0, 1.0)
    UNIFORM,        ///< Initialization to U(-0.05, 0.05)
    LECUN,          ///< Initialization to N(0, 1) * (1.0/sqrt(Nin))
    XAVIER,         ///< Initialization to N(0, 1) * (1.0/Nin))
    XAVIER_FULL,    ///< Initialization to N(0, 1) * (2.0/(Nin + Nout))
    HE              ///< Initialization to N(0, 1) * (2.0/sqrt(Nin))
};

} //end of dll namespace
