//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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
    void initialize_bias(B& b, size_t nin, size_t nout){
        cpp_unused(b);
        cpp_unused(nin);
        cpp_unused(nout);
    }

    template<typename W>
    void initialize_weights(W& w, size_t nin, size_t nout){
        cpp_unused(w);
        cpp_unused(nin);
        cpp_unused(nout);
    }
};

template<>
struct initializer_function<initializer_type::ZERO> {
    template<typename B>
    void initialize_bias(B& b, size_t nin, size_t nout){
        cpp_unused(nin);
        cpp_unused(nout);

        b = 0.0;
    }

    template<typename W>
    void initialize_weights(W& w, size_t nin, size_t nout){
        cpp_unused(nin);
        cpp_unused(nout);

        w = 0.0;
    }
};

} //end of dll namespace
