//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <vector>

namespace dll {

/*!
 * \brief Flatten the given vector of vector in a simple vector
 * \param deep The vector to flatten
 * \param flat The flatten vector
 */
template <typename One>
static void flatten_in(std::vector<std::vector<One>>& deep, std::vector<One>& flat) {
    flat.reserve(deep.size());

    for (auto& d : deep) {
        std::move(d.begin(), d.end(), std::back_inserter(flat));
    }
}

/*!
 * \brief Flatten the given vector of vector in a simple vector and
 * clear the input vector
 * \param deep The vector to flatten
 * \param flat The flatten vector
 */
template <typename One>
static void flatten_in_clr(std::vector<std::vector<One>>& deep, std::vector<One>& flat) {
    flat.reserve(deep.size());

    for (auto& d : deep) {
        std::move(d.begin(), d.end(), std::back_inserter(flat));
    }

    deep.clear();
}

/*!
 * \brief Flatten the given vector of vector and clear it
 * \param deep The vector to flatten
 * \return The flattened vector
 */
template <typename One>
static std::vector<One> flatten_clr(std::vector<std::vector<One>>& deep) {
    std::vector<One> flat;

    flatten_in_clr(deep, flat);

    return flat;
}

/*!
 * \brief Flatten the given vector of vector
 * \param deep The vector to flatten
 * \return The flattened vector
 */
template <typename One>
static std::vector<One> flatten(std::vector<std::vector<One>>& deep) {
    std::vector<One> flat;

    flatten_in(deep, flat);

    return flat;
}

} //end of dll namespace
