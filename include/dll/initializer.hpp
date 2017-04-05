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

template<initializer_type T>
struct initializer_function;

template<>
struct initializer_function<initializer_type::NONE> {
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(b);
        cpp_unused(nin);
        cpp_unused(nout);
    }
};

template<>
struct initializer_function<initializer_type::ZERO> {
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nin);
        cpp_unused(nout);

        b = etl::value_t<B>(0.0);
    }
};

template<>
struct initializer_function<initializer_type::GAUSSIAN> {
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nin);
        cpp_unused(nout);

        b = etl::normal_generator<etl::value_t<B>>(0.0, 1.0);
    }
};

template<>
struct initializer_function<initializer_type::SMALL_GAUSSIAN> {
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nin);
        cpp_unused(nout);

        b = etl::normal_generator<etl::value_t<B>>(0.0, 0.01);
    }
};

template<>
struct initializer_function<initializer_type::UNIFORM> {
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nin);
        cpp_unused(nout);

        b = etl::uniform_generator<etl::value_t<B>>(-0.05, 0.05);
    }
};

template<>
struct initializer_function<initializer_type::LECUN> {
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nout);

        b = etl::normal_generator<etl::value_t<B>>(0.0, 1.0) / sqrt(double(nin));
    }
};

template<>
struct initializer_function<initializer_type::XAVIER> {
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nout);

        b = etl::normal_generator<etl::value_t<B>>(0.0, 1.0) * sqrt(1.0 / nin);
    }
};

template<>
struct initializer_function<initializer_type::XAVIER_FULL> {
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        b = etl::normal_generator<etl::value_t<B>>(0.0, 1.0) * sqrt(2.0 / (nin + nout));
    }
};

template<>
struct initializer_function<initializer_type::HE> {
    template<typename B>
    static void initialize(B& b, size_t nin, size_t nout){
        cpp_unused(nout);

        b = etl::normal_generator<etl::value_t<B>>(0.0, 1.0) * sqrt(2.0 / nin);
    }
};

} //end of dll namespace
