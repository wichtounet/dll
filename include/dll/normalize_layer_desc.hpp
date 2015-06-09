//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_NORMALIZE_LAYER_DESC_HPP
#define DLL_NORMALIZE_LAYER_DESC_HPP

namespace dll {

struct normalize_layer_desc {
    /*! The layer type */
    using layer_t = normalize_layer<normalize_layer_desc>;
};

} //end of dll namespace

#endif
