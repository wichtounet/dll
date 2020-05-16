//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <atomic>
#include <thread>

namespace dll {

/*!
 * \brief Helper to create and initialize a cache for inputs
 *
 * The cache is for putting all the inputs inside.
 * The big cache is for storing several batches.
 */
template <typename Desc, typename Iterator, typename Enable = void>
struct cache_helper;

/*!
 * \brief cache_helper implementation for 1D inputs
 */
template <typename Desc, typename Iterator>
struct cache_helper<Desc, Iterator, std::enable_if_t<etl::is_1d<typename std::iterator_traits<Iterator>::value_type>>> {
    using T = etl::value_t<typename std::iterator_traits<Iterator>::value_type>; ///< Input type

    using cache_type     = etl::dyn_matrix<T, 2>; ///< The type of the cache
    using big_cache_type = etl::dyn_matrix<T, 3>; ///< The type of the big cache

    static constexpr size_t batch_size     = Desc::BatchSize;    ///< The size of the generated batches
    static constexpr size_t big_batch_size = Desc::BigBatchSize; ///< The number of batches kept in cache

    /*!
     * \brief Init the cache
     * \param n The size of the cache
     * \param it An iterator to an element
     * \param cache The cache to initialize
     */
    static void init(size_t n, const Iterator& it, cache_type& cache) {
        auto one = *it;
        cache    = cache_type(n, etl::dim<0>(one));
    }

    /*!
     * \brief Init the big cache
     * \param it An iterator to an element
     * \param cache The big cache to initialize
     */
    static void init_big(Iterator& it, big_cache_type& cache) {
        auto one = *it;
        cache    = big_cache_type(big_batch_size, batch_size, etl::dim<0>(one));
    }
};

/*!
 * \brief cache_helper implementation for 3D inputs
 */
template <typename Desc, typename Iterator>
struct cache_helper<Desc, Iterator, std::enable_if_t<etl::is_3d<typename std::iterator_traits<Iterator>::value_type>>> {
    using T = etl::value_t<typename std::iterator_traits<Iterator>::value_type>; ///< Input type

    using cache_type     = etl::dyn_matrix<T, 4>; ///< The type of the cache
    using big_cache_type = etl::dyn_matrix<T, 5>; ///< The type of the big cache

    static constexpr size_t batch_size     = Desc::BatchSize;    ///< The size of the generated batches
    static constexpr size_t big_batch_size = Desc::BigBatchSize; ///< The number of batches kept in cache

    /*!
     * \brief Init the cache
     * \param n The size of the cache
     * \param it An iterator to an element
     * \param cache The cache to initialize
     */
    static void init(size_t n, const Iterator& it, cache_type& cache) {
        auto one = *it;
        cache    = cache_type(n, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));
    }

    /*!
     * \brief Init the big cache
     * \param it An iterator to an element
     * \param cache The big cache to initialize
     */
    static void init_big(Iterator& it, big_cache_type& cache) {
        auto one = *it;

        if (Desc::random_crop_x && Desc::random_crop_y) {
            cache = big_cache_type(big_batch_size, batch_size, etl::dim<0>(one), Desc::random_crop_y, Desc::random_crop_x);
        } else {
            cache = big_cache_type(big_batch_size, batch_size, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));
        }
    }
};

/*!
 * \brief cache_helper implementation for 3D inputs
 */
template <typename Desc, typename Iterator>
struct cache_helper<Desc, Iterator, std::enable_if_t<etl::is_2d<typename std::iterator_traits<Iterator>::value_type>>> {
    using T = etl::value_t<typename std::iterator_traits<Iterator>::value_type>; ///< Input type

    using cache_type     = etl::dyn_matrix<T, 3>; ///< The type of the cache
    using big_cache_type = etl::dyn_matrix<T, 4>; ///< The type of the big cache

    static constexpr size_t batch_size     = Desc::BatchSize;    ///< The size of the generated batches
    static constexpr size_t big_batch_size = Desc::BigBatchSize; ///< The number of batches kept in cache

    /*!
     * \brief Init the cache
     * \param n The size of the cache
     * \param it An iterator to an element
     * \param cache The cache to initialize
     */
    static void init(size_t n, const Iterator& it, cache_type& cache) {
        auto one = *it;
        cache    = cache_type(n, etl::dim<0>(one), etl::dim<1>(one));
    }

    /*!
     * \brief Init the big cache
     * \param it An iterator to an element
     * \param cache The big cache to initialize
     */
    static void init_big(Iterator& it, big_cache_type& cache) {
        auto one = *it;

        if (Desc::random_crop_x && Desc::random_crop_y) {
            cache = big_cache_type(big_batch_size, batch_size, Desc::random_crop_y, Desc::random_crop_x);
        } else {
            cache = big_cache_type(big_batch_size, batch_size, etl::dim<0>(one), etl::dim<1>(one));
        }
    }
};

} //end of dll namespace
