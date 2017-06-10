//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cstddef>

#include "unit_type.hpp"
#include "function.hpp"
#include "loss.hpp"
#include "decay_type.hpp"
#include "lr_driver_type.hpp"
#include "sparsity_method.hpp"
#include "bias_mode.hpp"
#include "initializer.hpp"

namespace dll {

template <typename ID>
struct basic_conf_elt {
    using type_id = ID;
};

template <typename ID, typename T>
struct type_conf_elt {
    using type_id = ID;

    using value = T;
};

template <typename ID, template <typename...> class T>
struct template_type_conf_elt {
    using type_id = ID;

    template <typename RBM>
    using value = T<RBM>;
};

template <typename ID, template <typename, bool> class T>
struct template_type_tb_conf_elt {
    using type_id = ID;

    template <typename RBM, bool Denoising>
    using value = T<RBM, Denoising>;
};

template <typename ID, typename T, T value>
struct value_conf_elt : std::integral_constant<T, value> {
    using type_id = ID;
};

template <typename ID, typename T, T v1, T v2>
struct value_pair_conf_elt {
    using type_id = ID;

    static constexpr T value_1 = v1;
    static constexpr T value_2 = v2;
};

struct copy_id;
struct elastic_id;
struct batch_size_id;
struct big_batch_size_id;
struct visible_id;
struct hidden_id;
struct pooling_id;
struct activation_id;
struct loss_id;
struct initializer_id;
struct initializer_bias_id;
struct weight_decay_id;
struct lr_driver_id;
struct trainer_id;
struct trainer_rbm_id;
struct watcher_id;
struct sparsity_id;
struct bias_id;
struct momentum_id;
struct parallel_mode_id;
struct serial_id;
struct verbose_id;
struct horizontal_id;
struct vertical_id;
struct shuffle_id;
struct shuffle_pre_id;
struct svm_concatenate_id;
struct svm_scale_id;
struct init_weights_id;
struct clip_gradients_id;
struct weight_type_id;
struct free_energy_id;
struct no_epoch_error_id;
struct random_crop_id;
struct batch_mode_id;
struct dbn_only_id;
struct horizontal_mirroring_id;
struct vertical_mirroring_id;
struct categorical_id;
struct threaded_id;
struct nop_id;
struct elastic_distortion_id;
struct noise_id;
struct scale_pre_id;
struct normalize_pre_id;
struct binarize_pre_id;
struct autoencoder_id;

/*!
 * \brief Sets the minibatch size
 * \tparam B The minibatch size
 */
template <size_t B>
struct batch_size : value_conf_elt<batch_size_id, size_t, B> {};

/*!
 * \brief Sets the big batch size.
 *
 * This is the number of minibatch that the DBN will load at once.
 *
 * \tparam B The big batch size
 */
template <size_t B>
struct big_batch_size : value_conf_elt<big_batch_size_id, size_t, B> {};

/*!
 * \brief Sets the visible unit type
 * \tparam VT The visible unit type
 */
template <unit_type VT>
struct visible : value_conf_elt<visible_id, unit_type, VT> {};

/*!
 * \brief Sets the hidden unit type
 * \tparam HT The hidden unit type
 */
template <unit_type HT>
struct hidden : value_conf_elt<hidden_id, unit_type, HT> {};

/*!
 * \brief Sets the pooling unit type
 * \tparam PT The pooling unit type
 */
template <unit_type PT>
struct pooling : value_conf_elt<pooling_id, unit_type, PT> {};

/*!
 * \brief Sets the activation function
 * \tparam FT The activation function type
 */
template <function FT>
struct activation : value_conf_elt<activation_id, function, FT> {};

/*!
 * \brief Sets the loss function
 * \tparam FT The loss function type
 */
template <loss_function FT>
struct loss : value_conf_elt<loss_id, loss_function, FT> {};

/*!
 * \brief Sets the initializer
 * \tparam IT The initializer type
 */
template <initializer_type IT>
struct initializer : value_conf_elt<initializer_id, initializer_type, IT> {};

/*!
 * \brief Sets the initializer
 * \tparam IT The initializer type
 */
template <initializer_type IT>
struct initializer_bias : value_conf_elt<initializer_bias_id, initializer_type, IT> {};

/*!
 * \brief Enable and select weight decay
 * \tparam T The type of weight decay
 */
template <decay_type T = decay_type::L2>
struct weight_decay : value_conf_elt<weight_decay_id, decay_type, T> {};

/*!
 * \brief Set the learning rate driver
 */
template <lr_driver_type T = lr_driver_type::FIXED>
struct lr_driver : value_conf_elt<lr_driver_id, lr_driver_type, T> {};

/*!
 * \brief Copy augmentation
 */
template <size_t C>
struct copy : value_conf_elt<copy_id, size_t, C> {};

/*!
 * \brief Sets the random cropping size
 * \tparam B The minibatch size
 */
template <size_t X, size_t Y>
struct random_crop : value_pair_conf_elt<random_crop_id, size_t, X, Y> {};

/*!
 * \brief Elastic distortion
 */
template <size_t C, size_t K = 9>
struct elastic : basic_conf_elt<elastic_id> {};

/*!
 * \brief Activate sparsity and select the method to use
 */
template <sparsity_method M = sparsity_method::GLOBAL_TARGET>
struct sparsity : value_conf_elt<sparsity_id, sparsity_method, M> {};

/*!
 * \brief Select the bias method
 */
template <bias_mode M = bias_mode::SIMPLE>
struct bias : value_conf_elt<bias_id, bias_mode, M> {};

/*!
 * \brief Sets the type to use to store (and compute) the weights
 * \tparam The weight type
 */
template <typename T>
struct weight_type : type_conf_elt<weight_type_id, T> {};

/*!
 * \brief sets the trainer for DBN
 * \tparam The trainer type
 */
template <template <typename...> class T>
struct trainer : template_type_conf_elt<trainer_id, T> {};

/*!
 * \brief sets the trainer for RBM
 * \tparam The trainer type
 */
template <template <typename> class T>
struct trainer_rbm : template_type_conf_elt<trainer_rbm_id, T> {};

/*!
 * \brief sets the watcher
 * \tparam The watcher type
 */
template <template <typename...> class T>
struct watcher : template_type_conf_elt<watcher_id, T> {};

/*!
 * \brief Enable momentum learning
 */
struct momentum : basic_conf_elt<momentum_id> {};

/*
 * \brief Use parallel mode instead of batch mode
 */
struct parallel_mode : basic_conf_elt<parallel_mode_id> {};

/*!
 * \brief Disable threading
 */
struct serial : basic_conf_elt<serial_id> {};

/*!
 * \brief Make execution as verbose as possible
 */
struct verbose : basic_conf_elt<verbose_id> {};

/*!
 * \brief Concatenate the features of each layer for SVM training
 */
struct svm_concatenate : basic_conf_elt<svm_concatenate_id> {};

/*!
 * \brief Scale the features for SVM training
 */
struct svm_scale : basic_conf_elt<svm_scale_id> {};

/*!
 * \brief Use horizontal mirroring for data augmentation
 */
struct horizontal_mirroring : basic_conf_elt<horizontal_mirroring_id> {};

/*!
 * \brief Use vertical mirroring for data augmentation
 */
struct vertical_mirroring : basic_conf_elt<vertical_mirroring_id> {};

/*!
 * \brief Transform the labels into categorical matrix
 */
struct categorical : basic_conf_elt<categorical_id> {};

/*!
 * \brief Use a thread for data augmentation.
 */
struct threaded : basic_conf_elt<threaded_id> {};

/*!
 * \brief Sets the elastic distortion kernel
 * \tparam K The elastic distortion kernel
 */
template <size_t K>
struct elastic_distortion : value_conf_elt<elastic_distortion_id, size_t, K> {};

/*!
 * \brief Sets the noise
 * \tparam N The percent of noise
 */
template <size_t N>
struct noise : value_conf_elt<noise_id, size_t, N> {};

/*!
 * \brief Sets the prescaling factor
 * \tparam S The scaling factor
 */
template <size_t S>
struct scale_pre : value_conf_elt<scale_pre_id, size_t, S> {};

/*!
 * \brief Sets the binarize threshold
 * \tparam B The binarize threshold
 */
template <size_t B>
struct binarize_pre : value_conf_elt<binarize_pre_id, size_t, B> {};

/*!
 * \brief Normalize the inputs
 */
struct normalize_pre : basic_conf_elt<normalize_pre_id> {};

/*!
 * \brief Sets the mode to auto-encoder.
 */
struct autoencoder : basic_conf_elt<autoencoder_id> {};

/*!
 * \brief Initialize the weights of an RBM given the inputs
 */
struct init_weights : basic_conf_elt<init_weights_id> {};

/*!
 * \brief Shuffle the inputs before each epoch.
 */
struct shuffle : basic_conf_elt<shuffle_id> {};

/*!
 * \brief dbn: Shuffle the inputs before each pretraining epoch.
 * This implies that the inputs will be copied in memory!
 */
struct shuffle_pre : basic_conf_elt<shuffle_pre_id> {};

/*!
 * \brief Enable free energy computation
 */
struct free_energy : basic_conf_elt<free_energy_id> {};

/*!
 * \brief Disable error calculation on epoch.
 */
struct no_epoch_error : basic_conf_elt<no_epoch_error_id> {};

/*!
 * \brief Enable gradient clipping.
 */
struct clip_gradients : basic_conf_elt<clip_gradients_id> {};

/*!
 * \brief Indicates that the layer is only made to be used in a DBN.
 *
 * This will disable a few fields and save some memory
 */
struct dbn_only : basic_conf_elt<dbn_only_id> {};

/*!
 * \brief Do nothing (for TMP)
 */
struct nop : basic_conf_elt<nop_id> {};

/*!
 * \brief Use batch mode in DBN (Do not process the complete dataset at once)
 */
struct batch_mode : basic_conf_elt<batch_mode_id> {};

/*!
 * \brief Conditional shuffle (shuffle if Cond = true)
 */
template <bool Cond>
using shuffle_cond = std::conditional_t<Cond, shuffle, nop>;

/*!
 * \brief Conditional gradient clipping (clip gradients if Cond = true)
 */
template <bool Cond>
using clipping_cond = std::conditional_t<Cond, clip_gradients, nop>;

/*!
 * \brief Conditional pre normalization.
 */
template <bool Cond>
using normalize_pre_cond = std::conditional_t<Cond, normalize_pre, nop>;

/*!
 * \brief Conditional auto-encoder configuration.
 */
template <bool Cond>
using autoencoder_cond = std::conditional_t<Cond, autoencoder, nop>;

} //end of dll namespace
