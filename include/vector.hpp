//=======================================================================
// Copyright Baptiste Wicht 2014.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#ifndef DBN_VECTOR_HPP
#define DBN_VECTOR_HPP

template<typename T>
class vector {
private:
    const size_t rows;
    T* const _data;

public:
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
};

#endif
