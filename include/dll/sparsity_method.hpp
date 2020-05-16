//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Define how sparsity is applied
 */
enum class sparsity_method {
    NONE,          ///< Don't train a sparse representation
    GLOBAL_TARGET, ///< Sparsity according to (Nair and Hinton, 2009) but using global penalty
    LOCAL_TARGET,  ///< Sparsity according to (Nair and Hinton, 2009)
    LEE            ///< Sparsity according to (Lee, 2009)
};

} //end of dll namespace
