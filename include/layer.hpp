//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_LAYER_HPP
#define DBN_LAYER_HPP

namespace dbn {

template<std::size_t visibles, std::size_t hiddens>
struct layer {
    static constexpr const std::size_t num_visible = visibles;
    static constexpr const std::size_t num_hidden = hiddens;
};

} //end of dbn namespace

#endif