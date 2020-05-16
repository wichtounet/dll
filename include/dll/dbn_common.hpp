//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

namespace dbn_detail {

template <typename W, typename Enable = void>
struct rbm_watcher_type {
    using type = void;
};

template <typename W>
struct rbm_watcher_type<W, std::enable_if_t<W::replace_sub>> {
    using type = W;
};

template <typename W>
using rbm_watcher_t = typename rbm_watcher_type<W>::type;

} //end of namespace dbn_detail

} //end of namespace dll
