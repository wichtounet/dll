//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file blas.hpp
 * \brief Utility for using BLAS.
 */

#ifndef DLL_BLAS
#define DLL_BLAS

namespace dll {

#ifdef ETL_BLAS_MODE

inline void blas_ger(std::size_t n1, std::size_t n2, float* a, float* b, float* c){
    cblas_sger(CblasRowMajor, n1, n2, 1.0f, a, 1, b, 1, c, n2);
}

inline void blas_ger(std::size_t n1, std::size_t n2, double* a, double* b, double* c){
    cblas_dger(CblasRowMajor, n1, n2, 1.0, a, 1, b, 1, c, n2);
}

#endif //ETL_BLAS_MODE

} //end of dll namespace

#endif
