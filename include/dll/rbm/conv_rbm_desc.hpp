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
 * \brief Describe a Convolutional Restricted Boltzmann Machine.  *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 * Once configured, the ::layer_t member returns the type of the configured RBM.
 */
template <size_t NC_T, size_t NV_1, size_t NV_2, size_t K_T, size_t NW_1, size_t NW_2, typename... Parameters>
struct conv_rbm_desc {
    static constexpr size_t NV1 = NV_1; ///< The first dimension of the input
    static constexpr size_t NV2 = NV_2; ///< The second dimension of the input
    static constexpr size_t NW1 = NW_1; ///< The first dimension of the output
    static constexpr size_t NW2 = NW_2; ///< The second dimension of the output
    static constexpr size_t NC  = NC_T; ///< The number of input channels
    static constexpr size_t K   = K_T;  ///< The number of filters

    /*!
     * \brief A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*!
     * \brief The batch size for training this layer
     */
    static constexpr size_t BatchSize         = detail::get_value_v<batch_size<1>, Parameters...>;

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
    using layer_t = conv_rbm_impl<conv_rbm_desc<NC_T, NV1, NV2, K_T, NW1, NW2, Parameters...>>;

private:
    template <typename... Args>
    struct dyn_layer_t_impl {
        using sequence = remove_type_id<batch_size_id, Args...>;

        using type = typename build_dyn_layer_t<dyn_conv_rbm_impl, dyn_conv_rbm_desc, sequence, Args...>::type;
    };

public:
    /*!
     * The dynamic layer type
     */
    using dyn_layer_t = typename dyn_layer_t_impl<Parameters...>::type;

    //Validate all parameters

    static_assert(NV1 > 0, "A matrix of at least 1x1 is necessary for the visible units");
    static_assert(NV2 > 0, "A matrix of at least 1x1 is necessary for the visible units");
    static_assert(NW1 > 0, "A matrix of at least 1x1 is necessary for the kernel");
    static_assert(NW2 > 0, "A matrix of at least 1x1 is necessary for the kernel");
    static_assert(NC > 0, "At least one channel is necessary");
    static_assert(K > 0, "At least one group is necessary");
    static_assert(BatchSize > 0, "Batch size must be at least 1");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<
                             momentum_id, batch_size_id, visible_id, hidden_id, dbn_only_id,
                             weight_decay_id, sparsity_id, trainer_rbm_id, watcher_id, clip_gradients_id,
                             bias_id, weight_type_id, shuffle_id, verbose_id, nop_id>,
                         Parameters...>,
        "Invalid parameters type");

    static_assert(Sparsity == sparsity_method::NONE || hidden_unit == unit_type::BINARY,
                  "Sparsity only works with binary hidden units");
};

/*!
 * \brief Describe a Convolutional Restricted Boltzmann Machine with square inputs and filters.
 *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 * Once configured, the ::rbm_t member returns the type of the configured RBM.
 */
template <size_t NC_T, size_t NV_T, size_t K_T, size_t NW_T, typename... Parameters>
using conv_rbm_square_desc = conv_rbm_desc<NC_T, NV_T, NV_T, K_T, NW_T, NW_T, Parameters...>;

/*!
 * \brief Describe a Convolutional Restricted Boltzmann Machine.  *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 * Once configured, the ::layer_t member returns the type of the configured RBM.
 */
template <size_t NC_T, size_t NV_1, size_t NV_2, size_t K_T, size_t NW_1, size_t NW_2, typename... Parameters>
using conv_rbm = typename conv_rbm_desc<NC_T, NV_1, NV_2, K_T, NW_1, NW_2, Parameters...>::layer_t;

/*!
 * \brief Describe a Convolutional Restricted Boltzmann Machine with square inputs and filters.
 *
 * This struct should be used to define a RBM either as standalone or for a DBN.
 * Once configured, the ::rbm_t member returns the type of the configured RBM.
 */
template <size_t NC_T, size_t NV_T, size_t K_T, size_t NW_T, typename... Parameters>
using conv_rbm_square = typename conv_rbm_desc<NC_T, NV_T, NV_T, K_T, NW_T, NW_T, Parameters...>::layer_t;

} //end of dll namespace
