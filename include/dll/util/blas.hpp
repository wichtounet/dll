//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

inline void blas_ger(size_t n1, size_t n2, float alpha, float* a, float* b, float* c) {
    cblas_sger(CblasRowMajor, n1, n2, alpha, a, 1, b, 1, c, n2);
}

inline void blas_ger(size_t n1, size_t n2, double alpha, double* a, double* b, double* c) {
    cblas_dger(CblasRowMajor, n1, n2, alpha, a, 1, b, 1, c, n2);
}

inline void blas_axpy(size_t n1, float alpha, float* a, float* b) {
    cblas_saxpy(n1, alpha, a, 1, b, 1);
}

inline void blas_axpy(size_t n1, double alpha, double* a, double* b) {
    cblas_daxpy(n1, alpha, a, 1, b, 1);
}

#endif //ETL_BLAS_MODE

} //end of dll namespace

#endif
