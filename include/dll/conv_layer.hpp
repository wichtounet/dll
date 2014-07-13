//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONV_LAYER_HPP
#define DBN_CONV_LAYER_HPP

#include "base_conf.hpp"
#include "contrastive_divergence.hpp"
#include "watcher.hpp"
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
        is_valid<tmp_list<
                momentum, batch_size_id, visible_id, hidden_id,
                weight_decay_id>
            , Parameters...>::value,
        "Invalid parameters type");

    static constexpr const bool Momentum = is_present<momentum, Parameters...>::value;
    static constexpr const std::size_t BatchSize = get_value<batch_size<1>, Parameters...>::value;
    static constexpr const unit_type visible_unit = get_value<visible<unit_type::BINARY>, Parameters...>::value;
    static constexpr const unit_type hidden_unit = get_value<hidden<unit_type::BINARY>, Parameters...>::value;
    static constexpr const decay_type Decay = get_value<weight_decay<decay_type::NONE>, Parameters...>::value;

    template <typename RBM>
    using trainer_t = typename get_template_type<trainer<cd1_trainer_t>, Parameters...>::template type<RBM>;

    template <typename RBM>
    using watcher_t = typename get_template_type<watcher<default_watcher>, Parameters...>::template type<RBM>;

    static_assert(BatchSize > 0, "Batch size must be at least 1");

    using rbm_t = conv_rbm<conv_layer<NV_T, NH_T, K_T, Parameters...>>;
};

} //end of dbn namespace

#endif