//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_VECTOR_HPP
#define DBN_VECTOR_HPP

#include "assert.hpp"

template<typename T>
class vector {
private:
    size_t rows;
    T* _data;

public:
    typedef T value_type;
    typedef T* iterator;
    typedef const T* const_iterator;

    vector() : rows(0), _data(nullptr){
        //Nothing else to init
    }

    vector(size_t r) : rows(r), _data(new T[r]){
        //Nothing else to init
    }

    vector(size_t r, const T& value) : rows(r), _data(new T[r]){
        std::fill(_data, _data + size(), value);
    }

    vector(const vector& rhs) = delete;
    vector& operator=(const vector& rhs) = delete;

    vector(vector&& rhs) : rows(rhs.rows), _data(rhs._data) {
        rhs.rows = 0;
        rhs._data = nullptr;
    }

    vector& operator=(vector&& rhs){
        rows = rhs.rows;
        _data = rhs._data;

        rhs.rows = 0;
        rhs._data = nullptr;
    }

    ~vector(){
        delete[] _data;
    }

    size_t size() const {
        return rows;
    }

    void operator=(const T& value){
        std::fill(_data, _data + size(), value);
    }

    T& operator()(size_t i){
        dbn_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    const T& operator()(size_t i) const {
        dbn_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    const T* data() const {
        return _data;
    }

    const_iterator begin() const {
        return &_data[0];
    }

    iterator begin(){
        return &_data[0];
    }

    const_iterator end() const {
        return &_data[rows];
    }

    iterator end(){
        return &_data[rows];
    }
};

#endif
