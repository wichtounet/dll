//=======================================================================
// Copyright Baptiste Wicht 2014.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#ifndef DBN_MATRIX_HPP
#define DBN_MATRIX_HPP

template<typename T>
class matrix {
private:
    const size_t rows;
    const size_t columns;
    T* const _data;

public:
    matrix() : rows(0), columns(0), _data(nullptr) {
        //Nothing else to init
    }

    matrix(size_t r, size_t c) :
            rows(r), columns(c), _data(new T[r* c]){
        //Nothing else to init
    }

    matrix(size_t r, size_t c, const T& value) :
            rows(r), columns(c), _data(new T[r * c]){
        std::fill(_data, _data + size(), value);
    }

    matrix(const matrix& rhs) = delete;
    matrix& operator=(const matrix& rhs) = delete;

    ~matrix(){
        delete[] _data;
    }

    size_t size() const {
        return rows * columns;
    }

    void operator=(const T& value){
        std::fill(_data, _data + size(), value);
    }

    T& operator()(size_t i, size_t j){
        dbn_assert(i < rows, "Out of bounds");
        dbn_assert(j < columns, "Out of bounds");

        return _data[i * columns + j];
    }

    const T& operator()(size_t i, size_t j) const {
        dbn_assert(i < rows, "Out of bounds");
        dbn_assert(j < columns, "Out of bounds");

        return _data[i * columns + j];
    }

    const T* data() const {
        return _data;
    }
};


#endif
