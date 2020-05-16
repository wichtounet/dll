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
 * \brief Describe a dynamic Convolutional Restricted Boltzmann Machine.  *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 * Once configured, the ::layer_t member returns the type of the configured RBM.
 */
template <typename... Parameters>
struct dyn_conv_rbm_desc {
    /*!
     * \brief A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*!
     * \brief The batch size for training this layer
     */
    static constexpr size_t BatchSize    = detail::get_value_v<batch_size<25>, Parameters...>;

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
     * \brief The sparsity bias mode (LEE)
     */
    static constexpr bias_mode Bias           = detail::get_value_v<bias<bias_mode::SIMPLE>, Parameters...>;

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*! The type of the trainer to use to train the RBM */
    template <typename RBM>
    using trainer_t = typename detail::get_template_type<trainer_rbm<cd1_trainer_t>, Parameters...>::template value<RBM>;

    /*! The type of the watched to use during training */
    template <typename RBM>
    using watcher_t = typename detail::get_template_type<watcher<default_rbm_watcher>, Parameters...>::template value<RBM>;

    /*! The layer type */
    using layer_t = dyn_conv_rbm_impl<dyn_conv_rbm_desc<Parameters...>>;

    /*! The layer type */
    using dyn_layer_t = layer_t;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<
                             batch_size_id, momentum_id, visible_id, hidden_id, dbn_only_id, clip_gradients_id,
                             weight_decay_id, sparsity_id, trainer_rbm_id, watcher_id,
                             bias_id, weight_type_id, shuffle_id, verbose_id, nop_id>,
                         Parameters...>,
        "Invalid parameters type");

    static_assert(Sparsity == sparsity_method::NONE || hidden_unit == unit_type::BINARY,
                  "Sparsity only works with binary hidden units");
};

/*!
 * \brief Describe a dynamic Convolutional Restricted Boltzmann Machine.
 *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 */
template <typename... Parameters>
using dyn_conv_rbm = typename dyn_conv_rbm_impl<Parameters...>::layer_t;

} //end of dll namespace
