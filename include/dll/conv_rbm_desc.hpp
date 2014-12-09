//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_CONV_RBM_DESC_HPP
#define DLL_CONV_RBM_DESC_HPP

#include "base_conf.hpp"
#include "contrastive_divergence.hpp"
#include "watcher.hpp"
#include "tmp.hpp"

namespace dll {

/*!
 * \brief Describe a Convolutional Restricted Boltzmann Machine.
 *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 * Once configured, the ::rbm_t member returns the type of the configured RBM.
 */
template<std::size_t NV_T, std::size_t NC_T, std::size_t NH_T, std::size_t K_T, typename... Parameters>
struct conv_rbm_desc {
    static constexpr const std::size_t NV = NV_T;
    static constexpr const std::size_t NH = NH_T;
    static constexpr const std::size_t NC = NC_T;
    static constexpr const std::size_t K = K_T;

    static constexpr const bool Momentum = detail::is_present<momentum, Parameters...>::value;
    static constexpr const bool Parallel = detail::is_present<parallel, Parameters...>::value;
    static constexpr const std::size_t BatchSize = detail::get_value<batch_size<1>, Parameters...>::value;
    static constexpr const unit_type visible_unit = detail::get_value<visible<unit_type::BINARY>, Parameters...>::value;
    static constexpr const unit_type hidden_unit = detail::get_value<hidden<unit_type::BINARY>, Parameters...>::value;
    static constexpr const decay_type Decay = detail::get_value<weight_decay<decay_type::NONE>, Parameters...>::value;
    static constexpr const sparsity_method Sparsity = detail::get_value<sparsity<sparsity_method::NONE>, Parameters...>::value;
    static constexpr const bias_mode Bias = detail::get_value<bias<bias_mode::SIMPLE>, Parameters...>::value;
    static constexpr const bool Shuffle = detail::is_present<shuffle, Parameters...>::value;

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<double>, Parameters...>::type;

    /*! The type of the trainer to use to train the RBM */
    template <typename RBM>
    using trainer_t = typename detail::get_template_type<trainer<cd1_trainer_t>, Parameters...>::template type<RBM>;

    /*! The type of the watched to use during training */
    template <typename RBM>
    using watcher_t = typename detail::get_template_type<watcher<default_rbm_watcher>, Parameters...>::template type<RBM>;

    /*! The RBM type */
    using rbm_t = conv_rbm<conv_rbm_desc<NV_T, NC_T, NH_T, K_T, Parameters...>>;

    //Validate all parameters

    static_assert(NV > 0, "A matrix of at least 1x1 is necessary for the visible units");
    static_assert(NH > 0, "A matrix of at least 1x1 is necessary for the hidden units");
    static_assert(NC > 0, "At least one channel is necessary");
    static_assert(K > 0, "At least one group is necessary");
    static_assert(BatchSize > 0, "Batch size must be at least 1");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<detail::tmp_list<
                momentum_id, batch_size_id, visible_id, hidden_id,
                weight_decay_id, sparsity_id, trainer_id, watcher_id,
                bias_id, weight_type_id, shuffle_id, parallel_id>
            , Parameters...>::value,
        "Invalid parameters type");

    static_assert(Sparsity == sparsity_method::NONE || hidden_unit == unit_type::BINARY,
        "Sparsity only works with binary hidden units");
};

} //end of dll namespace

#endif
