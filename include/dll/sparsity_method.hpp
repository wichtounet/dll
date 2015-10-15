//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_SPARSITY_METHOD_HPP
#define DLL_SPARSITY_METHOD_HPP

namespace dll {

/*!
 * \brief Define how sparsity is applied
 */
enum class sparsity_method {
    NONE,          ///< Don't train a sparse representation
    GLOBAL_TARGET, ///< Sparsity according to (Nair and Hinton, 2009) but using global penalty
    LOCAL_TARGET,  ///< Sparsity according to (Nair and Hinton, 2009)
    LEE
};

} //end of dll namespace

#endif