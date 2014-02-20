//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_FAST_VECTOR_HPP
#define DBN_FAST_VECTOR_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <array>

#include "assert.hpp"
#include "fast_op.hpp"
#include "fast_expr.hpp"

template<typename T, std::size_t Rows>
class fast_vector {
private:
    std::array<T, Rows> _data;

public:
    static constexpr const std::size_t rows = Rows;

    typedef T value_type;
    typedef T* iterator;
    typedef const T* const_iterator;

    fast_vector(){
        //Nothing else to init
    }

    fast_vector(const T& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    //Construct from expression

    fast_vector& operator=(const fast_vector& rhs){
        for(std::size_t i = 0; i < Rows; ++i){
            _data[i] = rhs[i];
        }

        return *this;
    }

    template<typename LE, typename Op, typename RE>
    fast_vector(fast_expr<T, LE, Op, RE>&& e){
        for(std::size_t i = 0; i < Rows; ++i){
            _data[i] = e[i];
        }
    }

    template<typename LE, typename Op, typename RE>
    fast_vector& operator=(fast_expr<T, LE, Op, RE>&& e){
        for(std::size_t i = 0; i < Rows; ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    //Prohibit copy
    fast_vector(const fast_vector& rhs) = delete;

    //Allow move
    fast_vector(fast_vector&& rhs) = default;
    fast_vector& operator=(fast_vector&& rhs) = default;

    //Modifiers

    //Set every element to the same scalar
    void operator=(const T& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    //Multiply each element by a scalar
    fast_vector& operator*=(const T& value){
        for(size_t i = 0; i < Rows; ++i){
            _data[i] *= value;
        }

        return *this;
    }

    //Divide each element by a scalar
    fast_vector& operator/=(const T& value){
        for(size_t i = 0; i < Rows; ++i){
            _data[i] /= value;
        }

        return *this;
    }

    template<typename RE>
    fast_vector& operator+=(RE&& rhs){
        for(size_t i = 0; i < Rows; ++i){
            _data[i] += rhs[i];
        }

        return *this;
    }

    template<typename RE>
    fast_vector& operator-=(RE&& rhs){
        for(size_t i = 0; i < Rows; ++i){
            _data[i] -= rhs[i];
        }

        return *this;
    }

    //Add a scalar to each element
    template<typename RE, typename = enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator+(RE re) const -> fast_expr<T, const fast_vector&, plus_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    //Add elements of vector together
    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator+(RE&& re) const -> fast_expr<T, const fast_vector&, plus_binary_op<T>, decltype(std::forward<RE>(re))> {
        return {*this, std::forward<RE>(re)};
    }

    //Remove each element by a scalar
    template<typename RE, typename = enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator-(RE re) const -> fast_expr<T, const fast_vector&, minus_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    //Sub elements of vector together
    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator-(RE&& re) const -> fast_expr<T, const fast_vector&, minus_binary_op<T>, decltype(std::forward<RE>(re))> {
        return {*this, std::forward<RE>(re)};
    }

    //Mul each element by a scalar
    template<typename RE, typename = enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator*(RE re) const -> fast_expr<T, const fast_vector&, mul_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    //Mul elements of vector togethers
    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator*(RE&& re) const -> fast_expr<T, const fast_vector&, mul_binary_op<T>, decltype(std::forward<RE>(re))> {
        return {*this, std::forward<RE>(re)};
    }

    //Div each element by a scalar
    template<typename RE, typename = enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator/(RE re) const -> fast_expr<T, const fast_vector&, div_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    //Div elements of vector togethers
    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator/(RE&& re) const -> fast_expr<T, const fast_vector&, div_binary_op<T>, decltype(std::forward<RE>(re))> {
        return {*this, std::forward<RE>(re)};
    }

    //Accessors

    constexpr size_t size() const {
        return rows;
    }

    T& operator()(size_t i){
        dbn_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    const T& operator()(size_t i) const {
        dbn_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    T& operator[](size_t i){
        dbn_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    const T& operator[](size_t i) const {
        dbn_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    const T* data() const {
        return _data;
    }

    const_iterator begin() const {
        return _data.begin();
    }

    iterator begin(){
        return _data.begin();
    }

    const_iterator end() const {
        return _data.end();
    }

    iterator end(){
        return _data.end();
    }
};

#endif
