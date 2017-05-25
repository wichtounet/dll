//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <atomic>
#include <thread>

namespace dll {

template<typename Desc, typename Iterator, typename Enable = void>
struct cache_helper;

template<typename Desc, typename Iterator>
struct cache_helper<Desc, Iterator, std::enable_if_t<etl::is_1d<typename Iterator::value_type>::value>> {
    using T = etl::value_t<typename Iterator::value_type>;

    using cache_type = etl::dyn_matrix<T, 2>;
    using big_cache_type = etl::dyn_matrix<T, 3>;

    static void init(size_t n, Iterator& it, cache_type& cache){
        auto one = *it;
        cache = cache_type(n, etl::dim<0>(one));
    }

    static void init_big(size_t big, size_t n, Iterator& it, big_cache_type& cache){
        auto one = *it;
        cache = big_cache_type(big, n, etl::dim<0>(one));
    }
};

template<typename Desc, typename Iterator>
struct cache_helper<Desc, Iterator, std::enable_if_t<etl::is_3d<typename Iterator::value_type>::value>> {
    using T = etl::value_t<typename Iterator::value_type>;

    using cache_type = etl::dyn_matrix<T, 4>;
    using big_cache_type = etl::dyn_matrix<T, 5>;

    static void init(size_t n, Iterator& it, cache_type& cache){
        auto one = *it;
        cache = cache_type(n, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));
    }

    static void init_big(size_t big, size_t n, Iterator& it, big_cache_type& cache){
        auto one = *it;

        if(Desc::random_crop_x && Desc::random_crop_y){
            cache = big_cache_type(big, n, etl::dim<0>(one), Desc::random_crop_y, Desc::random_crop_x);
        } else {
            cache = big_cache_type(big, n, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));
        }
    }
};

} //end of dll namespace
