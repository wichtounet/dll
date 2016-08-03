//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <std::size_t T_T = 30>
struct binarize_layer_desc {
    static constexpr const std::size_t T = T_T;

    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = binarize_layer<binarize_layer_desc<T_T>>;
};

} //end of dll namespace
