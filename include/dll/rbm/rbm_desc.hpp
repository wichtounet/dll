//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_conf.hpp"
#include "dll/contrastive_divergence.hpp"
#include "dll/watcher.hpp"
#include "dll/util/tmp.hpp"

namespace dll {

/*!
 * \brief Describe a RBM.
 *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 * Once configured, the ::layer_t member returns the type of the configured RBM.
 */
template <std::size_t visibles, std::size_t hiddens, typename... Parameters>
struct rbm_desc {
    static constexpr std::size_t num_visible = visibles; ///< The number of visible units
    static constexpr std::size_t num_hidden  = hiddens;  ///< The number of hidden units

    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    static constexpr std::size_t BatchSize    = detail::get_value<batch_size<1>, Parameters...>::value;
    static constexpr unit_type visible_unit   = detail::get_value<visible<unit_type::BINARY>, Parameters...>::value;
    static constexpr unit_type hidden_unit    = detail::get_value<hidden<unit_type::BINARY>, Parameters...>::value;
    static constexpr sparsity_method Sparsity = detail::get_value<sparsity<sparsity_method::NONE>, Parameters...>::value;

    /*!
     * The type used to store the weights
     */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*!
     * The type of the trainer to use to train the RBM
     */
    template <typename RBM, bool Denoising>
    using trainer_t = typename detail::get_template_type_tb<trainer_rbm<cd1_trainer_t>, Parameters...>::template value<RBM, Denoising>;

    /*!
     * The type of the watched to use during training
     */
    template <typename RBM>
    using watcher_t = typename detail::get_template_type<watcher<default_rbm_watcher>, Parameters...>::template value<RBM>;

    static_assert(num_visible > 0, "There must be at least 1 visible unit");
    static_assert(num_hidden > 0, "There must be at least 1 hidden unit");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<momentum_id, parallel_mode_id, serial_id, verbose_id, batch_size_id, visible_id,
                                        hidden_id, weight_decay_id, init_weights_id, sparsity_id, trainer_rbm_id, watcher_id,
                                        weight_type_id, shuffle_id, free_energy_id, dbn_only_id, nop_id, clip_gradients_id>,
                         Parameters...>::value,
        "Invalid parameters type for rbm_desc");

    static_assert(BatchSize > 0, "Batch size must be at least 1");

    static_assert(Sparsity == sparsity_method::NONE || hidden_unit == unit_type::BINARY,
                  "Sparsity only works with binary hidden units");

    /*!
     * The layer type
     */
    using layer_t = rbm<rbm_desc<visibles, hiddens, Parameters...>>;

private:
    template <typename... Args>
    struct dyn_layer_t_impl {
        using sequence = remove_type_id<batch_size_id, Args...>;

        using type = typename build_dyn_layer_t<dyn_rbm, dyn_rbm_desc, sequence, Args...>::type;
    };

public:
    /*!
     * The layer type
     */
    using dyn_layer_t = typename dyn_layer_t_impl<Parameters...>::type;
};

} //end of dll namespace
