//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

/*!
 * \brief The default DBN trainer (Stochastic-Gradient Descent)
 */
template <typename DBN>
using default_dbn_trainer_t = sgd_trainer<DBN>;

/*!
 * \brief Describe a DBN *
 *
 * This struct should be used to define a DBN.
 * Once configured, the ::network_t member (or ::dbn_t) returns the type of the configured DBN.
 */
template <template <typename> typename DBN_T, typename Layers, typename... Parameters>
struct generic_dbn_desc {
    using layers      = Layers; ///< The network layers
    using base_layers = Layers; ///< The network layers before transformation

    /*!
     * \brief A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*!
     * \brief The batch size for training this layer
     */
    static constexpr size_t BatchSize    = detail::get_value_v<batch_size<1>, Parameters...>;

    /*!
     * \brief The number of batches kept in cache
     */
    static constexpr size_t BigBatchSize = detail::get_value_v<big_batch_size<1>, Parameters...>;

    /*!
     * \brief The pre scaling factor
     */
    static constexpr size_t ScalePre = detail::get_value_v<scale_pre<0>, Parameters...>;

    /*!
     * \brief The noise factor
     */
    static constexpr size_t Noise = detail::get_value_v<noise<0>, Parameters...>;

    /*!
     * \brief The pre binarization thresholding
     */
    static constexpr size_t BinarizePre = detail::get_value_v<binarize_pre<0>, Parameters...>;

    /*!
     * \brief Indicates if input are normalized
     */
    static constexpr bool NormalizePre = parameters::template contains<normalize_pre>();

    /*!
     * \brief The type of loss used for training
     */
    static constexpr auto Loss = detail::get_value_v<loss<loss_function::CATEGORICAL_CROSS_ENTROPY>, Parameters...>;

    /*!
     * \brief The type of updater for SGD
     */
    static constexpr auto Updater = detail::get_value_v<updater<updater_type::SGD>, Parameters...>;

    /*!
     * \brief The type of strategy for early stopping
     */
    static constexpr auto Early = detail::get_value_v<early_stopping<strategy::ERROR_GOAL>, Parameters...>;

    /*! The type of the trainer to use to train the DBN */
    template <typename DBN>
    using trainer_t = typename detail::get_template_type<trainer<default_dbn_trainer_t>, Parameters...>::template value<DBN>;

    /*! The type of the watched to use during training */
    template <typename DBN>
    using watcher_t = typename detail::get_template_type<watcher<default_dbn_watcher>, Parameters...>::template value<DBN>;

    using output_policy_t = detail::get_type_t<output_policy<default_output_policy>, Parameters...>; ///< The output policy

    /*! The DBN type */
    using dbn_t = DBN_T<generic_dbn_desc<DBN_T, Layers, Parameters...>>;

    /*!
     * \brief The network type.
     *
     * This is the same as the DBN type, only kept for legacy
     * reasons.
     */
    using network_t = dbn_t;

    static_assert(BatchSize > 0, "Batch size must be at least 1");
    static_assert(BigBatchSize > 0, "Big Batch size must be at least 1");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<
            cpp::type_list<
                trainer_id, watcher_id, weight_decay_id, big_batch_size_id, batch_size_id, verbose_id, 
                no_batch_display_id, no_epoch_error_id,
                batch_mode_id, svm_concatenate_id, svm_scale_id, serial_id, shuffle_id, shuffle_pre_id, loss_id,
                normalize_pre_id, binarize_pre_id, scale_pre_id, autoencoder_id, noise_id, updater_id,
                early_stopping_id, early_training_id, clip_gradients_id, output_policy_id>,
            Parameters...>,
        "Invalid parameters type");
};

} //end of dll namespace

// Include the trainers
#include "dll/trainer/conjugate_gradient.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"
