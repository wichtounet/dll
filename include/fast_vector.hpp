//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_FAST_VECTOR_HPP
#define DBN_FAST_VECTOR_HPP

#include "assert.hpp"

template<typename T, std::size_t Rows>
class fast_vector {
    static_assert(Rows > 0, "Vector of size 0 do no make sense");

private:
    T* _data;

public:
    static constexpr const std::size_t rows = Rows;

    typedef T value_type;
    typedef T* iterator;
    typedef const T* const_iterator;

    fast_vector() : _data(new T[rows]){
        //Nothing else to init
    }

    fast_vector(const T& value) : _data(new T[rows]){
        std::fill(_data, _data + size(), value);
    }

    //Prohibit copy
    fast_vector(const fast_vector& rhs) = delete;
    fast_vector& operator=(const fast_vector& rhs) = delete;

    //Allow move
    fast_vector(fast_vector&& rhs) : _data(rhs._data) {
        rhs._data = nullptr;
    }

    fast_vector& operator=(fast_vector&& rhs){
        _data = rhs._data;
        rhs._data = nullptr;
    }

    ~fast_vector(){
        delete[] _data;
    }

    //Modifiers

    void operator=(const T& value){
        std::fill(_data, _data + size(), value);
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
