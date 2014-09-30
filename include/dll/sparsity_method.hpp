//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
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
    TARGET         ///< Sparsity according to (Nair and Hinton, 2009)
};

} //end of dbn namespace

#endif