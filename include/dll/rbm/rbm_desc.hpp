//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
template <size_t visibles, size_t hiddens, typename... Parameters>
struct rbm_desc {
    static constexpr size_t num_visible = visibles; ///< The number of visible units
    static constexpr size_t num_hidden  = hiddens;  ///< The number of hidden units

    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*!
     * \brief The batch size for training this layer
     */
    static constexpr size_t BatchSize    = detail::get_value_v<batch_size<1>, Parameters...>;

    /*!
     * \brief The type of visible unit
     */
    static constexpr unit_type visible_unit   = detail::get_value_v<visible<unit_type::BINARY>, Parameters...>;

    /*!
     * \brief The type of hidden unit
     */
    static constexpr unit_type hidden_unit    = detail::get_value_v<hidden<unit_type::BINARY>, Parameters...>;

    /*!
     * \brief The sparsity penalty for pretraining
     */
    static constexpr sparsity_method Sparsity = detail::get_value_v<sparsity<sparsity_method::NONE>, Parameters...>;

    /*!
     * The type used to store the weights
     */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*!
     * The type of the trainer to use to train the RBM
     */
    template <typename RBM>
    using trainer_t = typename detail::get_template_type<trainer_rbm<cd1_trainer_t>, Parameters...>::template value<RBM>;

    /*!
     * The type of the watched to use during training
     */
    template <typename RBM>
    using watcher_t = typename detail::get_template_type<watcher<default_rbm_watcher>, Parameters...>::template value<RBM>;

    static_assert(num_visible > 0, "There must be at least 1 visible unit");
    static_assert(num_hidden > 0, "There must be at least 1 hidden unit");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<momentum_id, verbose_id, batch_size_id, visible_id,
                                        hidden_id, weight_decay_id, init_weights_id, sparsity_id, trainer_rbm_id, watcher_id,
                                        weight_type_id, shuffle_id, free_energy_id, dbn_only_id, nop_id, clip_gradients_id>,
                         Parameters...>,
        "Invalid parameters type for rbm_desc");

    static_assert(BatchSize > 0, "Batch size must be at least 1");

    static_assert(Sparsity == sparsity_method::NONE || hidden_unit == unit_type::BINARY,
                  "Sparsity only works with binary hidden units");

    /*!
     * The layer type
     */
    using layer_t = rbm_impl<rbm_desc<visibles, hiddens, Parameters...>>;

private:
    template <typename... Args>
    struct dyn_layer_t_impl {
        using sequence = remove_type_id<batch_size_id, Args...>;

        using type = typename build_dyn_layer_t<dyn_rbm_impl, dyn_rbm_desc, sequence, Args...>::type;
    };

public:
    /*!
     * The layer type
     */
    using dyn_layer_t = typename dyn_layer_t_impl<Parameters...>::type;
};

/*!
 * \brief Describe a RBM.
 *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 * Once configured, the ::layer_t member returns the type of the configured RBM.
 */
template <size_t visibles, size_t hiddens, typename... Parameters>
using rbm = typename rbm_desc<visibles, hiddens, Parameters...>::layer_t;

} //end of dll namespace
