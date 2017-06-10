//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Helper to create and initialize a cache for labels
 */
template <typename Desc, typename T, typename LIterator, typename Enable = void>
struct label_cache_helper;

/*!
 * \brief Helper to create and initialize a cache for labels.
 *
 * This version makes the label categorical.
 */
template <typename Desc, typename T, typename LIterator>
struct label_cache_helper<Desc, T, LIterator, std::enable_if_t<Desc::Categorical && !etl::is_etl_expr<typename std::iterator_traits<LIterator>::value_type>::value>> {
    using cache_type     = etl::dyn_matrix<T, 2>; ///< The type of the cache
    using big_cache_type = etl::dyn_matrix<T, 3>; ///< The type of the big cache

    /*!
     * \brief Init the cache
     * \param n The size of the cache
     * \param n_classes The number of classes
     * \param it An iterator to an element
     * \param cache The cache to initialize
     */
    static void init(size_t n, size_t n_classes, const LIterator& it, cache_type& cache) {
        cache = cache_type(n, n_classes);
        cache = T(0);

        cpp_unused(it);
    }

    /*!
     * \brief Init the big cache
     * \param big The number of batches
     * \param n The size of one batch
     * \param n_classes The number of classes
     * \param it An iterator to an element
     * \param cache The big cache to initialize
     */
    static void init_big(size_t big, size_t n, size_t n_classes, const LIterator& it, big_cache_type& cache) {
        cache = big_cache_type(big, n, n_classes);

        cpp_unused(it);
    }

    /*!
     * \brief Set the value of a label in the cache from the iterator
     * \param i The index of the label in the cache
     * \param it The label iterator
     * \param cache The label cache
     */
    template <typename E>
    static void set(size_t i, const LIterator& it, E&& cache) {
        cache(i) = T(0);
        cache(i, *it) = T(1);
    }
};

/*!
 * \brief Helper to create and initialize a cache for labels.
 *
 * This version keeps the flat label as such.
 */
template <typename Desc, typename T, typename LIterator>
struct label_cache_helper<Desc, T, LIterator, std::enable_if_t<!Desc::Categorical && !etl::is_etl_expr<typename std::iterator_traits<LIterator>::value_type>::value>> {
    using cache_type     = etl::dyn_matrix<T, 1>; ///< The type of the cache
    using big_cache_type = etl::dyn_matrix<T, 2>; ///< The type of the big cache

    /*!
     * \brief Init the cache
     * \param n The size of the cache
     * \param n_classes The number of classes
     * \param it An iterator to an element
     * \param cache The cache to initialize
     */
    static void init(size_t n, size_t n_classes, const LIterator& it, cache_type& cache) {
        cache = cache_type(n);

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    /*!
     * \brief Init the big cache
     * \param big The number of batches
     * \param n The size of one batch
     * \param n_classes The number of classes
     * \param it An iterator to an element
     * \param cache The big cache to initialize
     */
    static void init_big(size_t big, size_t n, size_t n_classes, const LIterator& it, big_cache_type& cache) {
        cache = big_cache_type(big, n);

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    /*!
     * \brief Set the value of a label in the cache from the iterator
     * \param i The index of the label in the cache
     * \param it The label iterator
     * \param cache The label cache
     */
    template <typename E>
    static void set(size_t i, const LIterator& it, E&& cache) {
        cache[i] = *it;
    }
};

/*!
 * \brief Helper to create and initialize a cache for labels.
 *
 * This version keeps the ETL 1D label as such.
 */
template <typename Desc, typename T, typename LIterator>
struct label_cache_helper<Desc, T, LIterator, std::enable_if_t<etl::is_1d<typename std::iterator_traits<LIterator>::value_type>::value>> {
    using cache_type     = etl::dyn_matrix<T, 2>; ///< The type of the cache
    using big_cache_type = etl::dyn_matrix<T, 3>; ///< The type of the big cache

    static_assert(!Desc::Categorical, "Cannot make such vector labels categorical");

    /*!
     * \brief Init the cache
     * \param n The size of the cache
     * \param n_classes The number of classes
     * \param it An iterator to an element
     * \param cache The cache to initialize
     */
    static void init(size_t n, size_t n_classes, const LIterator& it, cache_type& cache) {
        auto one = *it;
        cache    = cache_type(n, etl::dim<0>(one));

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    /*!
     * \brief Init the big cache
     * \param big The number of batches
     * \param n The size of one batch
     * \param n_classes The number of classes
     * \param it An iterator to an element
     * \param cache The big cache to initialize
     */
    static void init_big(size_t big, size_t n, size_t n_classes, const LIterator& it, big_cache_type& cache) {
        auto one = *it;
        cache    = big_cache_type(big, n, etl::dim<0>(one));

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    /*!
     * \brief Set the value of a label in the cache from the iterator
     * \param i The index of the label in the cache
     * \param it The label iterator
     * \param cache The label cache
     */
    template <typename E>
    static void set(size_t i, const LIterator& it, E&& cache) {
        cache(i) = *it;
    }
};

/*!
 * \brief Helper to create and initialize a cache for labels.
 *
 * This version keeps the ETL 3D label as such.
 */
template <typename Desc, typename T, typename LIterator>
struct label_cache_helper<Desc, T, LIterator, std::enable_if_t<etl::is_3d<typename std::iterator_traits<LIterator>::value_type>::value>> {
    using cache_type     = etl::dyn_matrix<T, 4>; ///< The type of the cache
    using big_cache_type = etl::dyn_matrix<T, 5>; ///< The type of the big cache

    static_assert(!Desc::Categorical, "Cannot make such matrix labels categorical");

    /*!
     * \brief Init the cache
     * \param n The size of the cache
     * \param n_classes The number of classes
     * \param it An iterator to an element
     * \param cache The cache to initialize
     */
    static void init(size_t n, size_t n_classes, const LIterator& it, cache_type& cache) {
        auto one = *it;
        cache    = cache_type(n, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    /*!
     * \brief Init the big cache
     * \param big The number of batches
     * \param n The size of one batch
     * \param n_classes The number of classes
     * \param it An iterator to an element
     * \param cache The big cache to initialize
     */
    static void init_big(size_t big, size_t n, size_t n_classes, const LIterator& it, big_cache_type& cache) {
        auto one = *it;
        cache    = big_cache_type(big, n, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));

        cpp_unused(it);
        cpp_unused(n_classes);
    }

    /*!
     * \brief Set the value of a label in the cache from the iterator
     * \param i The index of the label in the cache
     * \param it The label iterator
     * \param cache The label cache
     */
    template <typename E>
    static void set(size_t i, const LIterator& it, E&& cache) {
        cache(i) = *it;
    }
};

} //end of dll namespace
