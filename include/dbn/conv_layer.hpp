//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONV_LAYER_HPP
#define DBN_CONV_LAYER_HPP

namespace dbn {

template<typename C, std::size_t NV_T, std::size_t NH_T, std::size_t K_T>
struct conv_layer {
    static constexpr const std::size_t NV = NV_T;
    static constexpr const std::size_t NH = NH_T;
    static constexpr const std::size_t K = K_T;

    typedef C Conf;
};

} //end of dbn namespace

#endif