//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_LABELS_HPP
#define DBN_LABELS_HPP

#include <vector>

namespace dbn {

template<typename V>
struct fake_label_array {
    V value;

    fake_label_array(V v) : value(v) {}

    double operator[](size_t i) const {
        if(i == value){
            return 1.0;
        } else {
            return 0.0;
        }
    }
};

template<typename T>
typename std::vector<fake_label_array<T>> make_fake(const std::vector<T>& values){
    std::vector<fake_label_array<T>> fake;
    fake.reserve(values.size());

    for(auto v: values){
        fake.emplace_back(v);
    }

    return std::move(fake);
}

} //end of dbn namespace

#endif