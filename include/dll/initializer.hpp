//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "initializer_type.hpp"

/*!
 * \brief Initialization methods
 */

#pragma once

namespace dll {

/*!
 * \brief Functor for initializer functions
 * \tparam T The type of the initialization function
 */
template<initializer_type T>
struct initializer_function;

/*!
 * \brief Initialization function no-op
 */
template<>
struct initializer_function<initializer_type::NONE> {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(b);
        cpp_unused(nin);
        cpp_unused(nout);
    }
};

/*!
 * \brief Initialization function to zero
 */
template<>
struct initializer_function<initializer_type::ZERO> {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nin);
        cpp_unused(nout);

        b = etl::value_t<B>(0.0);
    }
};

/*!
 * \brief Initialization function to normal distribution
 */
template<>
struct initializer_function<initializer_type::GAUSSIAN> {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nin);
        cpp_unused(nout);

        b = etl::normal_generator<etl::value_t<B>>(0.0, 1.0);
    }
};

/*!
 * \brief Initialization function to small gaussian distribution
 */
template<>
struct initializer_function<initializer_type::SMALL_GAUSSIAN> {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nin);
        cpp_unused(nout);

        b = etl::normal_generator<etl::value_t<B>>(0.0, 0.01);
    }
};

/*!
 * \brief Initialization function to small uniform distribution
 */
template<>
struct initializer_function<initializer_type::UNIFORM> {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nin);
        cpp_unused(nout);

        b = etl::uniform_generator<etl::value_t<B>>(-0.05, 0.05);
    }
};

/*!
 * \brief Initialization function according to Lecun
 */
template<>
struct initializer_function<initializer_type::LECUN> {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nout);

        b = etl::normal_generator<etl::value_t<B>>(0.0, 1.0) / sqrt(double(nin));
    }
};

/*!
 * \brief Initialization function according to Xavier
 */
template<>
struct initializer_function<initializer_type::XAVIER> {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nout);

        b = etl::normal_generator<etl::value_t<B>>(0.0, 1.0) * sqrt(1.0 / nin);
    }
};

/*!
 * \brief Initialization function according to Xavier (with fanin+fanout)
 */
template<>
struct initializer_function<initializer_type::XAVIER_FULL> {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        b = etl::normal_generator<etl::value_t<B>>(0.0, 1.0) * sqrt(2.0 / (nin + nout));
    }
};

/*!
 * \brief Initialization function according to He
 */
template<>
struct initializer_function<initializer_type::HE> {
    /*!
     * \brief Initialize the given weights (or biases) according
     * to the initialization function
     * \param b The weights or biases to initialize
     * \param nin The neurons input
     * \param nin The neurons output
     */
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nout);

        b = etl::normal_generator<etl::value_t<B>>(0.0, 1.0) * sqrt(2.0 / nin);
    }
};

} //end of dll namespace
