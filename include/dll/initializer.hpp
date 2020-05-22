//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/util/random.hpp"

/*!
 * \brief Initialization methods
 */

#pragma once

namespace dll {

/*!
 * \brief Initialization function no-op
 */
struct init_none {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize([[maybe_unused]] B& b, [[maybe_unused]] size_t nin, [[maybe_unused]] size_t nout){
        // Nothing to init
    }
};

/*!
 * \brief Initialization function to zero
 */
template<typename Ratio>
struct init_constant {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, [[maybe_unused]] size_t nin, [[maybe_unused]] size_t nout){
        b = etl::value_t<B>(Ratio::num) / etl::value_t<B>(Ratio::den);
    }
};

/*!
 * \brief Initialize all the values to 0
 */
using init_zero = init_constant<std::ratio<0, 1>>;

/*!
 * \brief Initialize all the values to 1
 */
using init_one = init_constant<std::ratio<1, 1>>;

/*!
 * \brief Initialization function to normal distribution
 */
template<typename Mean = std::ratio<0, 1>, typename Std = std::ratio<1, 1>>
struct init_normal {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template <typename B>
    static void initialize(B& b, [[maybe_unused]] size_t nin, [[maybe_unused]] size_t nout) {
        constexpr auto mean   = etl::value_t<B>(Mean::num) / etl::value_t<B>(Mean::den);
        constexpr auto stddev = etl::value_t<B>(Std::num) / etl::value_t<B>(Std::den);

        b = etl::normal_generator<etl::value_t<B>>(dll::rand_engine(), mean, stddev);
    }
};

/*!
 * \brief Initialization function to uniform distribution
 */
template<typename A = std::ratio<-5, 100>, typename B = std::ratio<5, 100>>
struct init_uniform {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param w The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename W>
    static void initialize(W& w, [[maybe_unused]] size_t nin, [[maybe_unused]] size_t nout){
        constexpr auto a = etl::value_t<W>(A::num) / etl::value_t<W>(A::den);
        constexpr auto b = etl::value_t<W>(B::num) / etl::value_t<W>(B::den);

        w = etl::uniform_generator<etl::value_t<W>>(dll::rand_engine(), a, b);
    }
};

#define constant(f) std::ratio<(std::intmax_t)(f * 1'000'000), 1'000'000>

/*!
 * \brief Initialization function according to Lecun
 */
struct init_lecun {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, [[maybe_unused]] size_t nout){
        b = etl::normal_generator<etl::value_t<B>>(dll::rand_engine(), 0.0, 1.0) / sqrt(double(nin));
    }
};

/*!
 * \brief Initialization function according to Xavier
 */
struct init_xavier {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, [[maybe_unused]] size_t nout){
        b = etl::normal_generator<etl::value_t<B>>(dll::rand_engine(), 0.0, 1.0) * sqrt(1.0 / nin);
    }
};

/*!
 * \brief Initialization function according to Xavier (with fanin+fanout)
 */
struct init_xavier_full {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        b = etl::normal_generator<etl::value_t<B>>(dll::rand_engine(), 0.0, 1.0) * sqrt(2.0 / (nin + nout));
    }
};

/*!
 * \brief Initialization function according to He
 */
struct init_he {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, [[maybe_unused]] size_t nout){
        b = etl::normal_generator<etl::value_t<B>>(dll::rand_engine(), 0.0, 1.0) * sqrt(2.0 / nin);
    }
};

} //end of dll namespace
