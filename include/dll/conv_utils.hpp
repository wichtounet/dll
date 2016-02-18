//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_CONV_UTILS_HPP
#define DLL_CONV_UTILS_HPP

#include "base_conf.hpp" //The configuration helpers
#include "rbm_base.hpp"  //The base class
#include "layer_traits.hpp"
#include "checks.hpp"

namespace dll {

constexpr const bool conv_multi_fast = etl::is_cblas_enabled || etl::is_cublas_enabled;

template <typename V, typename K, typename C>
static void conv_2d_multi(V&& v, K&& kernels, C&& features) {
    if (conv_multi_fast) {
        static constexpr const std::size_t v1 = etl::decay_traits<V>::template dim<0>();
        static constexpr const std::size_t v2 = etl::decay_traits<V>::template dim<1>();
        static constexpr const std::size_t k1 = etl::decay_traits<K>::template dim<0>();
        static constexpr const std::size_t k2 = etl::decay_traits<K>::template dim<1>();
        static constexpr const std::size_t k3 = etl::decay_traits<K>::template dim<2>();
        static constexpr const std::size_t F1 = etl::decay_traits<C>::template dim<0>();
        static constexpr const std::size_t F2 = etl::decay_traits<C>::template dim<1>();
        static constexpr const std::size_t F3 = etl::decay_traits<C>::template dim<2>();

        etl::fast_dyn_matrix<etl::value_t<K>, k2 * k3, (v1 - k2 + 1) * (v2 - k3 + 1)> input_col; //Rearranged input
        etl::fast_dyn_matrix<etl::value_t<K>, k1, k3, k2> prepared_k;                            //Transposed kernels
        etl::fast_dyn_matrix<etl::value_t<K>, F1, F3, F2> features_t;                            //Transposed features

        //Note: Here, we do not need to fflip because in definition of the formula, the weights are flipped

        for (std::size_t i = 0; i < k1; ++i) {
            prepared_k(i) = transpose(kernels(i));
        }

        im2col_direct(input_col, v, k3, k2);

        *mul(
            etl::reshape<F1, k3 * k2>(prepared_k),
            input_col,
            etl::reshape<F1, F2 * F3>(features_t));

        for (std::size_t k = 0; k < k1; ++k) {
            features(k) = transpose(features_t(k));
        }
    } else {
        //Standard version
        for (size_t k = 0; k < etl::dim<0>(kernels); ++k) {
            features(k) = etl::conv_2d_valid(v, kernels(k));
        }
    }
}

} //end of dll namespace

#endif
