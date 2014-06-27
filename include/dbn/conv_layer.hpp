//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONV_LAYER_HPP
#define DBN_CONV_LAYER_HPP

#include "base_conf.hpp"
#include "utils.hpp"
#include "tmp.hpp"

namespace dll {

template<std::size_t NV_T, std::size_t NH_T, std::size_t K_T, typename... Parameters>
struct conv_layer {
    static constexpr const std::size_t NV = NV_T;
    static constexpr const std::size_t NH = NH_T;
    static constexpr const std::size_t K = K_T;

    static_assert(NV > 0, "A matrix of at least 1x1 is necessary for the visible units");
    static_assert(NH > 0, "A matrix of at least 1x1 is necessary for the hidden units");
    static_assert(K > 0, "At least one group is necessary");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        is_valid<tmp_list<momentum, batch_size_id, visible_unit_id, hidden_unit_id>, Parameters...>::value,
        "Invalid parameters type");

    static constexpr const bool Momentum = is_present<momentum, Parameters...>::value;
    static constexpr const std::size_t BatchSize = get_value<batch_size<1>, Parameters...>::value;
    static constexpr const Type VisibleUnit = get_value<visible_unit<Type::SIGMOID>, Parameters...>::value;
    static constexpr const Type HiddenUnit = get_value<visible_unit<Type::SIGMOID>, Parameters...>::value;
};

} //end of dbn namespace

#endif