//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_LAYER_HPP
#define DBN_LAYER_HPP

#include "base_conf.hpp"
#include "utils.hpp"
#include "tmp.hpp"

namespace dll {

template <typename RBM>
using cd1_trainer_t = cd_trainer<1, RBM>;

template<std::size_t visibles, std::size_t hiddens, typename... Parameters>
struct layer {
    static constexpr const std::size_t num_visible = visibles;
    static constexpr const std::size_t num_hidden = hiddens;

    static_assert(num_visible > 0, "There must be at least 1 visible unit");
    static_assert(num_hidden > 0, "There must be at least 1 hidden unit");

    static constexpr const bool Momentum = is_present<momentum, Parameters...>::value;
    static constexpr const std::size_t BatchSize = get_value<batch_size<1>, Parameters...>::value;
    static constexpr const Type VisibleUnit = get_value<visible_unit<Type::SIGMOID>, Parameters...>::value;
    static constexpr const Type HiddenUnit = get_value<hidden_unit<Type::SIGMOID>, Parameters...>::value;
    static constexpr const DecayType Decay = get_value<weight_decay<DecayType::NONE>, Parameters...>::value;
    static constexpr const bool Init = is_present<init_weights, Parameters...>::value;
    static constexpr const bool DBN = is_present<in_dbn, Parameters...>::value;
    static constexpr const bool Debug = is_present<debug, Parameters...>::value;
    static constexpr const bool Sparsity = is_present<sparsity, Parameters...>::value;

    template <typename RBM>
    using trainer_t = typename get_template_type<trainer<cd1_trainer_t>, Parameters...>::template type<RBM>;
};

} //end of dbn namespace

#endif