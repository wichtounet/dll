//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_SCALE_LAYER_DESC_HPP
#define DLL_SCALE_LAYER_DESC_HPP

namespace dll {

template <int A_T, int B_T>
struct scale_layer_desc {
    static constexpr const int A = A_T;
    static constexpr const int B = B_T;

    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = scale_layer<scale_layer_desc<A, B>>;
};

} //end of dll namespace

#endif
