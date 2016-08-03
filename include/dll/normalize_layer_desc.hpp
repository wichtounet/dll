//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

struct normalize_layer_desc {
    using parameters = cpp::type_list<>;

    /*! The layer type */
    using layer_t = normalize_layer<normalize_layer_desc>;
};

} //end of dll namespace
