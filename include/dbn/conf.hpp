//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONF_HPP
#define DBN_CONF_HPP

#include <cstddef>

#include "contrastive_divergence.hpp"
#include "unit_type.hpp"
#include "decay_type.hpp"

namespace dll {

template <typename RBM>
using cd1_trainer_t = cd_trainer<1, RBM>;

template<bool M = true, std::size_t B = 1, bool I = true, bool DB = false, DecayType WD = DecayType::NONE, bool S = false,
         Type VT = Type::SIGMOID, Type HT = Type::SIGMOID, template<typename> class T = cd1_trainer_t, bool D = false>
struct conf {
    static constexpr const bool Momentum = M;
    static constexpr const std::size_t BatchSize = B;
    static constexpr const bool Init = I;
    static constexpr const Type VisibleUnit = VT;
    static constexpr const Type HiddenUnit = HT;
    static constexpr const bool DBN = DB;
    static constexpr const DecayType Decay = WD;
    static constexpr const bool Debug = D;
    static constexpr const bool Sparsity = S;

    template <typename RBM>
    using trainer_t = T<RBM>;
};

} //end of dbn namespace

#endif