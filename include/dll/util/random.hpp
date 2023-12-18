//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <random>

namespace dll {

/*
 * \brief Lehmer random number generator
 */
struct lehmer64_generator {
    using result_type = uint64_t;

    __uint128_t state;

    static inline uint64_t split_seed(uint64_t index) {
        uint64_t z = index + 0x9E3779B97F4A7C15UL;
        z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z          = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    lehmer64_generator(uint64_t seed) {
        state = (((__uint128_t)split_seed(seed)) << 64) + split_seed(seed + 1);
    }

    inline result_type operator()(){
        state *= 0xda942042e4dd58b5UL;
        return state >> 64;
    }

    static constexpr result_type max() {
        return std::numeric_limits<result_type>::max();
    }

    static constexpr result_type min() {
        return std::numeric_limits<result_type>::min();
    }
};

/*!
 * \brief The random engine used by the library
 */
using random_engine = lehmer64_generator;

namespace detail {

/*!
 * \brief Set or get the random seed of DLL
 * \param new_seed If not zero, set the seed to the new value
 * \return The current DLL random seed (or the updated one)
 */
inline size_t seed_impl(size_t new_seed = 0){
    static std::random_device rd;
    static size_t seed = rd();

    if(new_seed){
        seed = new_seed;
    }

    return seed;
}

} // end of namespace detail

/*!
 * \brief Return the seed of the DLL random generator
 * \return the DLL random seed
 */
inline size_t seed(){
    return detail::seed_impl();
}

/*!
 * \brief Set the seed of the DLL
 * \param new_seed The new seed (cannot be zero)
 */
inline void set_seed(size_t new_seed){
    detail::seed_impl(new_seed);
}

/*!
 * \brief Return a reference to the DLL random engine
 * \return The DLL random engine
 */
inline random_engine& rand_engine(){
    static random_engine engine(seed());

    return engine;
}

} //end of dll namespace
