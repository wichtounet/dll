//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_MATRIX_HPP
#define DBN_MATRIX_HPP

#include "assert.hpp"
#include "fast_op.hpp"
#include "fast_expr.hpp"

//TODO Ensure that the fast_expr that is taken comes from a matrix
//or least from a vector of Rows * Columns size

template<typename T, size_t Rows, size_t Columns>
class fast_matrix {
private:
    std::array<T, Rows * Columns> _data;

public:
    static constexpr const std::size_t rows = Rows;
    static constexpr const std::size_t columns = Columns;

    fast_matrix(){
        //Nothing to init
    }

    fast_matrix(const T& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    //Construct from expression

    template<typename LE, typename Op, typename RE>
    fast_matrix(fast_expr<T, LE, Op, RE>&& e){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename LE, typename Op, typename RE>
    fast_matrix& operator=(fast_expr<T, LE, Op, RE>&& e){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    //Prohibit copy
    fast_matrix(const fast_matrix& rhs) = delete;
    fast_matrix& operator=(const fast_matrix& rhs) = delete;

    //Make sure
    fast_matrix(fast_matrix&& rhs) = default;
    fast_matrix& operator=(fast_matrix&& rhs) = default;

    //Modifiers

    //Set the same value to each element of the matrix
    void operator=(const T& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    template<typename RE>
    fast_matrix& operator+=(RE&& rhs){
        for(size_t i = 0; i < size(); ++i){
            _data[i] += rhs[i];
        }

        return *this;
    }

    //Add a scalar to each element
    auto operator+(T re) const -> fast_expr<T, const fast_matrix&, plus_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    //Add elements of matrix together
    template<typename RE>
    auto operator+(RE&& re) const -> fast_expr<T, const fast_matrix&, plus_binary_op<T>, decltype(std::forward<RE>(re))> {
        return {*this, std::forward<RE>(re)};
    }

    //Remove each element by a scalar
    auto operator-(T re) const -> fast_expr<T, const fast_matrix&, minus_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    //Sub elements of matrix together
    template<typename RE>
    auto operator-(RE&& re) const -> fast_expr<T, const fast_matrix&, minus_binary_op<T>, decltype(std::forward<RE>(re))> {
        return {*this, std::forward<RE>(re)};
    }

    //Mul each element by a scalar
    auto operator*(T re) const -> fast_expr<T, const fast_matrix&, mul_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    //Mul elements of matrix togethers
    template<typename RE>
    auto operator*(RE&& re) const -> fast_expr<T, const fast_matrix&, mul_binary_op<T>, decltype(std::forward<RE>(re))> {
        return {*this, std::forward<RE>(re)};
    }

    //Div each element by a scalar
    auto operator/(T re) const -> fast_expr<T, const fast_matrix&, div_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    //Div elements of matrix togethers
    template<typename RE>
    auto operator/(RE&& re) const -> fast_expr<T, const fast_matrix&, div_binary_op<T>, decltype(std::forward<RE>(re))> {
        return {*this, std::forward<RE>(re)};
    }

    //Accessors

    constexpr size_t size() const {
        return Rows * Columns;
    }

    T& operator()(size_t i, size_t j){
        dbn_assert(i < Rows, "Out of bounds");
        dbn_assert(j < Columns, "Out of bounds");

        return _data[i * Columns + j];
    }

    const T& operator()(size_t i, size_t j) const {
        dbn_assert(i < Rows, "Out of bounds");
        dbn_assert(j < Columns, "Out of bounds");

        return _data[i * Columns + j];
    }

    const T& operator[](size_t i) const {
        dbn_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    T& operator[](size_t i){
        dbn_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    const T* data() const {
        return _data;
    }
};


#endif
