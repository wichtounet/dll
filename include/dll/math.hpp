//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_MATH_HPP
#define DBN_MATH_HPP

#include <cmath>

namespace dll {

template<typename W>
constexpr W logistic_sigmoid(W x){
    return 1.0 / (1.0 + std::exp(-x));
}

template<typename W>
constexpr W softplus(W x){
    return std::log(1.0 + std::exp(x));
}

} //end of dbn namespace

#endif