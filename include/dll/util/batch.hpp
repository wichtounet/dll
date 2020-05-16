//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iterator>

#include "cpp_utils/assert.hpp"

namespace dll {

/*!
 * \brief A batch of samples or labels
 */
template <typename Iterator>
struct batch {
    using size_type  = size_t;                                      ///< The size type of the batch
    using value_type = typename std::decay_t<Iterator>::value_type; ///< The value type of the batch

    Iterator first; ///< The iterator to the first element
    Iterator last;  ///< The iterator to the past the end element

    /*!
     * \brief Create a batch
     * \param first The first element of the batch
     * \param first The past-the-end element of the batch
     */
    batch(Iterator&& first, Iterator&& last)
            : first(std::forward<Iterator>(first)),
              last(std::forward<Iterator>(last)) {
        cpp_assert(std::distance(first, last) > 0, "Batch cannot be empty or reversed");
    }

    /*!
     * \brief Return an iterator pointing to the first element of the batch
     */
    Iterator begin() const {
        return first;
    }

    /*!
     * \brief Return an iterator pointing to the past-the-end element of the batch
     */
    Iterator end() const {
        return last;
    }

    /*!
     * \brief Return the size of the batch
     */
    size_type size() const {
        return std::distance(begin(), end());
    }
};

/*!
 * \brief Create a new a batch from the given iterators
 * \param first The first element of the batch
 * \param first The past-the-end element of the batch
 * \return the created batch around the given iterators
 */
template <typename Iterator>
batch<Iterator> make_batch(Iterator&& first, Iterator&& last) {
    return {std::forward<Iterator>(first), std::forward<Iterator>(last)};
}

} //end of dll namespace
