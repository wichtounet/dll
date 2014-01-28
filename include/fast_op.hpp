//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_FAST_OP_HPP
#define DBN_FAST_OP_HPP

template<typename T>
struct scalar {
    const T value;
    constexpr scalar(T v) : value(v) {}

    constexpr const T operator[](std::size_t) const {
        return value;
    }
};

template<typename T>
struct plus_binary_op {
    static constexpr T apply(const T& lhs, const T& rhs){
        return lhs + rhs;
    }
};

template<typename T>
struct minus_binary_op {
    static constexpr T apply(const T& lhs, const T& rhs){
        return lhs - rhs;
    }
};

template<typename T>
struct mul_binary_op {
    static constexpr T apply(const T& lhs, const T& rhs){
        return lhs * rhs;
    }
};

template<typename T>
struct div_binary_op {
    static constexpr T apply(const T& lhs, const T& rhs){
        return lhs / rhs;
    }
};

#endif
