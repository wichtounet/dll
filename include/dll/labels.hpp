//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_LABELS_HPP
#define DLL_LABELS_HPP

#include <iterator>
#include <vector>

namespace dll {

template<typename V>
struct fake_label_array {
    using value_type = V;
    using this_type = fake_label_array<value_type>;

    value_type value;

    fake_label_array(value_type v) : value(v) {}

    double operator[](std::size_t i) const {
        if(i == value){
            return 1.0;
        } else {
            return 0.0;
        }
    }
};

template<typename Iterator>
std::vector<fake_label_array<typename std::iterator_traits<Iterator>::value_type>> make_fake(Iterator first, Iterator last){
    std::vector<fake_label_array<typename std::iterator_traits<Iterator>::value_type>> fake;
    fake.reserve(std::distance(first, last));

    std::for_each(first, last, [&fake](auto v){
        fake.emplace_back(v);
    });

    return std::move(fake);
}

} //end of dll namespace

#endif