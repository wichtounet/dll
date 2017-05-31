//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template<typename Desc, typename T, typename LIterator, typename Enable = void>
struct label_cache_helper;

// Case 1: make the labels categorical
template<typename Desc, typename T, typename LIterator>
struct label_cache_helper<Desc, T, LIterator, std::enable_if_t<Desc::Categorical && !etl::is_etl_expr<typename std::iterator_traits<LIterator>::value_type>::value>> {
    using cache_type = etl::dyn_matrix<T, 2>;
    using big_cache_type = etl::dyn_matrix<T, 3>;

    static void init(size_t n, size_t n_classes, const LIterator& it, cache_type& cache){
        cache = cache_type(n, n_classes);
        cache = T(0);

        cpp_unused(it);
    }

    static void init_big(size_t big, size_t n, size_t n_classes, const LIterator& it, big_cache_type& cache){
        cache = big_cache_type(big, n, n_classes);

        cpp_unused(it);
    }

    template<typename E>
    static void set(size_t i, const LIterator& it, cache_type& cache){
        cache(i, *it) = T(1);
    }

    template<typename E>
    static void set(size_t i, const LIterator& it, E&& cache){
        cache(i) = T(0);
        cache(i, *it) = T(1);
    }
};

// Case 2: Keep the labels as 1D
// Note: May be useless
template<typename Desc, typename T, typename LIterator>
struct label_cache_helper<Desc, T, LIterator, std::enable_if_t<!Desc::Categorical && !etl::is_etl_expr<typename std::iterator_traits<LIterator>::value_type>::value>> {
    using cache_type = etl::dyn_matrix<T, 1>;
    using big_cache_type = etl::dyn_matrix<T, 2>;

    static void init(size_t n, size_t n_classes, const LIterator& it, cache_type& cache){
        cache = cache_type(n);

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    static void init_big(size_t big, size_t n, size_t n_classes, const LIterator& it, big_cache_type& cache){
        cache = big_cache_type(big, n);

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    template<typename E>
    static void set(size_t i, const LIterator& it, E&& cache){
        cache[i] = *it;
    }
};

// Case 3: 1D labels
template<typename Desc, typename T, typename LIterator>
struct label_cache_helper<Desc, T, LIterator, std::enable_if_t<etl::is_1d<typename std::iterator_traits<LIterator>::value_type>::value>> {
    using cache_type = etl::dyn_matrix<T, 2>;
    using big_cache_type = etl::dyn_matrix<T, 3>;

    static_assert(!Desc::Categorical, "Cannot make such vector labels categorical");

    static void init(size_t n, size_t n_classes, const LIterator& it, cache_type& cache){
        auto one = *it;
        cache = cache_type(n, etl::dim<0>(one));

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    static void init_big(size_t big, size_t n, size_t n_classes, const LIterator& it, big_cache_type& cache){
        auto one = *it;
        cache = big_cache_type(big, n, etl::dim<0>(one));

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    template<typename E>
    static void set(size_t i, const LIterator& it, E&& cache){
        cache(i) = *it;
    }
};

// Case 4: 3D labels
template<typename Desc, typename T, typename LIterator>
struct label_cache_helper<Desc, T, LIterator, std::enable_if_t<etl::is_3d<typename std::iterator_traits<LIterator>::value_type>::value>> {
    using cache_type = etl::dyn_matrix<T, 4>;
    using big_cache_type = etl::dyn_matrix<T, 5>;

    static_assert(!Desc::Categorical, "Cannot make such matrix labels categorical");

    static void init(size_t n, size_t n_classes, const LIterator& it, cache_type& cache){
        auto one = *it;
        cache = cache_type(n, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    static void init_big(size_t big, size_t n, size_t n_classes, const LIterator& it, big_cache_type& cache){
        auto one = *it;
        cache = big_cache_type(big, n, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    template<typename E>
    static void set(size_t i, const LIterator& it, E&& cache){
        cache(i) = *it;
    }
};

} //end of dll namespace
