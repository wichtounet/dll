//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONV_CONF_HPP
#define DBN_CONV_CONF_HPP

#include <cstddef>

#include "unit_type.hpp"

namespace dbn {

template<bool M = true, std::size_t B = 1, Type VT = Type::SIGMOID, Type HT = Type::SIGMOID>
struct conv_conf {
    static constexpr const bool Momentum = M;
    static constexpr const std::size_t BatchSize = B;
    static constexpr const Type VisibleUnit = VT;
    static constexpr const Type HiddenUnit = HT;
};

} //end of dbn namespace

#endif