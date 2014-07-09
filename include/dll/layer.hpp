//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_LAYER_HPP
#define DBN_LAYER_HPP

#include "base_conf.hpp"
#include "contrastive_divergence.hpp"
#include "tmp.hpp"

namespace dll {

template<std::size_t visibles, std::size_t hiddens, typename... Parameters>
struct layer {
    static constexpr const std::size_t num_visible = visibles;
    static constexpr const std::size_t num_hidden = hiddens;

    static_assert(num_visible > 0, "There must be at least 1 visible unit");
    static_assert(num_hidden > 0, "There must be at least 1 hidden unit");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        is_valid<tmp_list<momentum, batch_size_id, visible_id, hidden_id, weight_decay_id,
              init_weights, in_dbn, debug, sparsity, trainer_id>, Parameters...>::value,
        "Invalid parameters type");

    static constexpr const bool Momentum = is_present<momentum, Parameters...>::value;
    static constexpr const std::size_t BatchSize = get_value<batch_size<1>, Parameters...>::value;
    static constexpr const unit_type visible_unit = get_value<visible<unit_type::BINARY>, Parameters...>::value;
    static constexpr const unit_type hidden_unit = get_value<hidden<unit_type::BINARY>, Parameters...>::value;
    static constexpr const decay_type Decay = get_value<weight_decay<decay_type::NONE>, Parameters...>::value;
    static constexpr const bool Init = is_present<init_weights, Parameters...>::value;
    static constexpr const bool DBN = is_present<in_dbn, Parameters...>::value;
    static constexpr const bool Debug = is_present<debug, Parameters...>::value;
    static constexpr const bool Sparsity = is_present<sparsity, Parameters...>::value;

    template <typename RBM>
    using trainer_t = typename get_template_type<trainer<cd1_trainer_t>, Parameters...>::template type<RBM>;

    static_assert(BatchSize > 0, "Batch size must be at least 1");

    static_assert(!Sparsity || (Sparsity && hidden_unit == unit_type::BINARY),
        "Sparsity only works with binary hidden units");
};

} //end of dbn namespace

#endif