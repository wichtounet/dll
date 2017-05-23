//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "base_conf.hpp"
#include "watcher.hpp"
#include "util/tmp.hpp"

namespace dll {

template <typename DBN>
struct sgd_trainer;

template <typename DBN>
using default_dbn_trainer_t = sgd_trainer<DBN>;

/*!
 * \brief Describe a DBN *
 *
 * This struct should be used to define a DBN.
 * Once configured, the ::dbn_t member returns the type of the configured DBN.
 */
template <template <typename> class DBN_T, typename Layers, typename... Parameters>
struct generic_dbn_desc {
    using layers      = Layers; ///< The network layers
    using base_layers = Layers; ///< The network layers before transformation

    /*!
     * \brief A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    static constexpr size_t BatchSize    = detail::get_value<batch_size<1>, Parameters...>::value;
    static constexpr size_t BigBatchSize = detail::get_value<big_batch_size<1>, Parameters...>::value;

    static constexpr auto Loss = detail::get_value<loss<loss_function::CATEGORICAL_CROSS_ENTROPY>, Parameters...>::value;

    /*! The type of the trainer to use to train the DBN */
    template <typename DBN>
    using trainer_t = typename detail::get_template_type<trainer<default_dbn_trainer_t>, Parameters...>::template value<DBN>;

    /*! The type of the watched to use during training */
    template <typename DBN>
    using watcher_t = typename detail::get_template_type<watcher<default_dbn_watcher>, Parameters...>::template value<DBN>;

    /*! The DBN type */
    using dbn_t = DBN_T<generic_dbn_desc<DBN_T, Layers, Parameters...>>;

    static_assert(BatchSize > 0, "Batch size must be at least 1");
    static_assert(BigBatchSize > 0, "Big Batch size must be at least 1");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<
            cpp::type_list<
                trainer_id, watcher_id, momentum_id, weight_decay_id, big_batch_size_id, batch_size_id, verbose_id, no_epoch_error_id,
                batch_mode_id, svm_concatenate_id, svm_scale_id, serial_id, lr_driver_id, shuffle_id, shuffle_pre_id, loss_id>,
            Parameters...>::value,
        "Invalid parameters type");
};

} //end of dll namespace

// Include the trainers
#include "dll/trainer/conjugate_gradient.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"
