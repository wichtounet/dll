//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONF_HPP
#define DBN_CONF_HPP

#include <cstddef>

namespace dbn {

enum class Type {
    SIGMOID,
    EXP
};

template<bool M = true, std::size_t B = 1, bool I = true, bool DB = false, bool WD = true, Type T = Type::SIGMOID, bool D = false>
struct conf {
    static constexpr const bool Momentum = M;
    static constexpr const std::size_t BatchSize = B;
    static constexpr const bool Init = I;
    static constexpr const Type Unit = T;
    static constexpr const bool Debug = D;
    static constexpr const bool DBN = DB;
    static constexpr const bool Decay = WD;
};

} //end of dbn namespace

#endif