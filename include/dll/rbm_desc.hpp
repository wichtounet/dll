//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_RBM_DESC_HPP
#define DLL_RBM_DESC_HPP

#include "base_conf.hpp"
#include "contrastive_divergence.hpp"
#include "watcher.hpp"
#include "tmp.hpp"

namespace dll {

/*!
 * \brief Describe a RBM.
 *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 * Once configured, the ::rbm_t member returns the type of the configured RBM.
 */
template<std::size_t visibles, std::size_t hiddens, typename... Parameters>
struct rbm_desc {
    static constexpr const std::size_t num_visible = visibles;
    static constexpr const std::size_t num_hidden = hiddens;

    using parameters = cpp::type_list<Parameters...>;

    static constexpr const std::size_t BatchSize = detail::get_value<batch_size<1>, Parameters...>::value;
    static constexpr const unit_type visible_unit = detail::get_value<visible<unit_type::BINARY>, Parameters...>::value;
    static constexpr const unit_type hidden_unit = detail::get_value<hidden<unit_type::BINARY>, Parameters...>::value;
    static constexpr const sparsity_method Sparsity = detail::get_value<sparsity<sparsity_method::NONE>, Parameters...>::value;

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*! The type of the trainer to use to train the RBM */
    template <typename RBM, bool Denoising>
    using trainer_t = typename detail::get_template_type_tb<trainer_rbm<cd1_trainer_t>, Parameters...>::template value<RBM, Denoising>;

    /*! The type of the watched to use during training */
    template <typename RBM>
    using watcher_t = typename detail::get_template_type<watcher<default_rbm_watcher>, Parameters...>::template value<RBM>;

    /*! The RBM type */
    using rbm_t = rbm<rbm_desc<visibles, hiddens, Parameters...>>;

    static_assert(num_visible > 0, "There must be at least 1 visible unit");
    static_assert(num_hidden > 0, "There must be at least 1 hidden unit");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<momentum_id, parallel_id, verbose_id, batch_size_id, visible_id, hidden_id, weight_decay_id,
              init_weights_id, sparsity_id, trainer_rbm_id, watcher_id, weight_type_id, shuffle_id, free_energy_id>, Parameters...>::value,
        "Invalid parameters type for rbm_desc");

    static_assert(BatchSize > 0, "Batch size must be at least 1");

    static_assert(Sparsity == sparsity_method::NONE || hidden_unit == unit_type::BINARY,
        "Sparsity only works with binary hidden units");
};

} //end of dll namespace

#endif
