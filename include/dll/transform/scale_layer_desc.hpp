//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <int A_T, int B_T>
struct scale_layer_desc {
    static constexpr const int A = A_T;
    static constexpr const int B = B_T;

    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = scale_layer<scale_layer_desc<A, B>>;

    /*! The dynamic layer type */
    using dyn_layer_t = scale_layer<scale_layer_desc<A, B>>;
};

} //end of dll namespace
