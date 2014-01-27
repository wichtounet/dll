//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_MATRIX_HPP
#define DBN_MATRIX_HPP

#include "assert.hpp"

template<typename T, size_t Rows, size_t Columns>
class matrix {
private:
    std::array<T, Rows * Columns> _data;

public:
    matrix(){
        //Nothing to init
    }

    matrix(const T& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    matrix(const matrix& rhs) = delete;
    matrix& operator=(const matrix& rhs) = delete;

    matrix(matrix&& rhs) = default;
    matrix& operator=(matrix&& rhs) = default;

    size_t size() const {
        return Rows * Columns;
    }

    void operator=(const T& value){
        std::fill(_data.begin(), _data.end(), value);
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

    const T* data() const {
        return _data;
    }
};


#endif
