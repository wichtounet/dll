//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <random>

namespace dll {

/*!
 * \brief The random engine used by the library
 */
using random_engine = etl::random_engine;

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
