//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <rectifier_method M = rectifier_method::ABS>
struct rectifier_layer_desc {
    static constexpr const rectifier_method method = M;

    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = rectifier_layer<rectifier_layer_desc<M>>;
};

} //end of dll namespace
