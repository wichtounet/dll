//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_BATCH_HPP
#define DLL_BATCH_HPP

#include <iterator>

#include "cpp_utils/assert.hpp"

namespace dll {

template<typename Iterator>
struct batch {
    using size_type = std::size_t;

    Iterator first;
    Iterator last;

    batch(Iterator&& first, Iterator&& last):
            first(std::forward<Iterator>(first)),
            last(std::forward<Iterator>(last)){
        cpp_assert(std::distance(first, last) > 0, "Batch cannot be empty or reversed");
    }

    Iterator begin() const {
        return first;
    }

    Iterator end() const {
        return last;
    }

    size_type size() const {
        return std::distance(begin(), end());
    }
};

template<typename Iterator>
batch<Iterator> make_batch(Iterator&& first, Iterator&& last){
    return {std::forward<Iterator>(first), std::forward<Iterator>(last)};
}

} //end of dbn namespace

#endif