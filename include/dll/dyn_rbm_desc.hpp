//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DYN_RBM_DESC_HPP
#define DLL_DYN_RBM_DESC_HPP

#include "base_conf.hpp"
#include "contrastive_divergence.hpp"
#include "watcher.hpp"
#include "tmp.hpp"

namespace dll {

/*!
 * \brief Describe a dyn RBM.
 *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 * Once configured, the ::rbm_t member returns the type of the configured RBM.
 */
template<typename... Parameters>
struct dyn_rbm_desc {
    static constexpr const bool Momentum = detail::is_present<momentum, Parameters...>::value;
    static constexpr const unit_type visible_unit = detail::get_value<visible<unit_type::BINARY>, Parameters...>::value;
    static constexpr const unit_type hidden_unit = detail::get_value<hidden<unit_type::BINARY>, Parameters...>::value;
    static constexpr const decay_type Decay = detail::get_value<weight_decay<decay_type::NONE>, Parameters...>::value;
    static constexpr const bool Init = detail::is_present<init_weights, Parameters...>::value;
    static constexpr const sparsity_method Sparsity = detail::get_value<sparsity<sparsity_method::NONE>, Parameters...>::value;
    static constexpr const bool Shuffle = detail::is_present<shuffle, Parameters...>::value;

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::type;

    /*! The type of the trainer to use to train the RBM */
    template <typename RBM>
    using trainer_t = typename detail::get_template_type<trainer<cd1_trainer_t>, Parameters...>::template type<RBM>;

    /*! The type of the watched to use during training */
    template <typename RBM>
    using watcher_t = typename detail::get_template_type<watcher<default_rbm_watcher>, Parameters...>::template type<RBM>;

    /*! The RBM type */
    using rbm_t = dyn_rbm<dyn_rbm_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<detail::tmp_list<momentum_id, visible_id, hidden_id, weight_decay_id,
              init_weights_id, sparsity_id, trainer_id, weight_type_id, shuffle_id>, Parameters...>::value,
        "Invalid parameters type");

    static_assert(Sparsity == sparsity_method::NONE || hidden_unit == unit_type::BINARY,
        "Sparsity only works with binary hidden units");
};

} //end of dbn namespace

#endif
