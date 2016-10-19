//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

    static constexpr const unit_type visible_unit   = detail::get_value<visible<unit_type::BINARY>, Parameters...>::value;
    static constexpr const unit_type hidden_unit    = detail::get_value<hidden<unit_type::BINARY>, Parameters...>::value;
    static constexpr const sparsity_method Sparsity = detail::get_value<sparsity<sparsity_method::NONE>, Parameters...>::value;
    static constexpr const bias_mode Bias           = detail::get_value<bias<bias_mode::SIMPLE>, Parameters...>::value;

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*! The type of the trainer to use to train the RBM */
    template <typename RBM, bool Denoising>
    using trainer_t = typename detail::get_template_type_tb<trainer_rbm<cd1_trainer_t>, Parameters...>::template value<RBM, Denoising>;

    /*! The type of the watched to use during training */
    template <typename RBM>
    using watcher_t = typename detail::get_template_type<watcher<default_rbm_watcher>, Parameters...>::template value<RBM>;

    /*! The layer type */
    using layer_t = dyn_conv_rbm<dyn_conv_rbm_desc<Parameters...>>;

    /*! The layer type */
    using dyn_layer_t = dyn_conv_rbm<dyn_conv_rbm_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<
                             momentum_id, visible_id, hidden_id, dbn_only_id, memory_id,
                             weight_decay_id, sparsity_id, trainer_rbm_id, watcher_id,
                             bias_id, weight_type_id, shuffle_id, parallel_mode_id, serial_id, verbose_id, nop_id>,
                         Parameters...>::value,
        "Invalid parameters type");

    static_assert(Sparsity == sparsity_method::NONE || hidden_unit == unit_type::BINARY,
                  "Sparsity only works with binary hidden units");
};

} //end of dll namespace
