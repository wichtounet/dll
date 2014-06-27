//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_BATCH_HPP
#define DBN_BATCH_HPP

#include <utility>
#include <vector>

#include "assert.hpp"

namespace dll {

template<typename T>
struct batch {
    typedef typename std::vector<T>::const_iterator const_iterator_t;
    typedef typename std::size_t size_type;

    std::pair<const_iterator_t, const_iterator_t> values;

    batch(const_iterator_t&& it, const_iterator_t&& end): values(std::forward<const_iterator_t>(it), std::forward<const_iterator_t>(end)){
        dll_assert(std::distance(it, end) > 0, "Batch cannot be empty or reversed");
    }

    const_iterator_t begin() const {
        return values.first;
    }

    const_iterator_t end() const {
        return values.second;
    }

    size_type size() const {
        return std::distance(begin(), end());
    }

    const T& operator[](size_t i) const {
        return *(begin() + i);
    }
};

} //end of dbn namespace

#endif