//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_BINARIZE_LAYER_DESC_HPP
#define DLL_BINARIZE_LAYER_DESC_HPP

namespace dll {

template <std::size_t T_T = 30>
struct binarize_layer_desc {
    static constexpr const std::size_t T = T_T;

    /*! The layer type */
    using layer_t = binarize_layer<binarize_layer_desc<T_T>>;
};

} //end of dll namespace

#endif
