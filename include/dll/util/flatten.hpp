//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_FLATTEN_HPP
#define DLL_FLATTEN_HPP

#include <vector>

namespace dll {

template <typename One>
static void flatten_in(std::vector<std::vector<One>>& deep, std::vector<One>& flat) {
    flat.reserve(deep.size());

    for (auto& d : deep) {
        std::move(d.begin(), d.end(), std::back_inserter(flat));
    }
}

template <typename One>
static void flatten_in_clr(std::vector<std::vector<One>>& deep, std::vector<One>& flat) {
    flat.reserve(deep.size());

    for (auto& d : deep) {
        std::move(d.begin(), d.end(), std::back_inserter(flat));
    }

    deep.clear();
}

template <typename One>
static std::vector<One> flatten_clr(std::vector<std::vector<One>>& deep) {
    std::vector<One> flat;

    flatten_in_clr(deep, flat);

    return flat;
}

template <typename One>
static std::vector<One> flatten(std::vector<std::vector<One>>& deep) {
    std::vector<One> flat;

    flatten_in(deep, flat);

    return flat;
}

} //end of dll namespace

#endif
