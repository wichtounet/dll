//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "util/tmp.hpp"
#include "decay_type.hpp"

namespace dll {

/*!
 * \brief Type Traits to get information on the network type
 */
template <typename Network>
struct dbn_traits {
    using network_t = Network;
    using desc      = typename network_t::desc; ///< The descriptor of the layer

    /*!
     * \brief Indicates if the network is convolutional
     */
    static constexpr bool is_convolutional() noexcept {
        return desc::layers::is_convolutional;
    }

    /*!
     * \brief Indicates if the network is dynamic
     */
    static constexpr bool is_dynamic() noexcept {
        return desc::layers::is_dynamic;
    }

    /*!
     * \brief Get the updater type of the network.
     */
    static constexpr updater_type updater() noexcept {
        return desc::Updater;
    }

    /*!
     * \brief Indicates if the network runs in batch mode
     */
    static constexpr bool batch_mode() noexcept {
        return desc::parameters::template contains<dll::batch_mode>();
    }

    /*!
     * \brief Indicates if the network computes error on epoch.
     */
    static constexpr bool error_on_epoch() noexcept {
        return !desc::parameters::template contains<dll::no_epoch_error>();
    }

    /*!
     * \brief Indicates if early stopping strategy is forced to use
     * training statistics when validation statistics are available.
     */
    static constexpr bool early_uses_training() noexcept {
        return desc::parameters::template contains<dll::early_training>();
    }

    /*!
     * \brief Indicates if the network shuffles the inputs before each
     * fine-tuning epoch.
     */
    static constexpr bool shuffle() noexcept {
        return desc::parameters::template contains<dll::shuffle>();
    }

    /*!
     * \brief Indicates if the network shuffles the inputs before
     * each pretraining epoch in batch mode.
     */
    static constexpr bool shuffle_pretrain() noexcept {
        return desc::parameters::template contains<dll::shuffle_pre>();
    }

    /*!
     * \brief Indicates if the network features are concatenated from all levels
     */
    static constexpr bool concatenate() noexcept {
        return desc::parameters::template contains<svm_concatenate>();
    }

    /*!
     * \brief Indicates if the network cannot use threading
     */
    static constexpr bool is_serial() noexcept {
        return desc::parameters::template contains<serial>();
    }

    /*!
     * \brief Indicates if the network is verbose
     */
    static constexpr bool is_verbose() noexcept {
        return desc::parameters::template contains<verbose>();
    }

    /*!
     * \brief Indicates if the network is verbose
     */
    static constexpr bool should_display_batch() noexcept {
        return !desc::parameters::template contains<no_batch_display>();
    }

    /*!
     * \brief Indicates if the network scales its features before sending to SVM.
     */
    static constexpr bool scale() noexcept {
        return desc::parameters::template contains<svm_scale>();
    }

    /*!
     * \brief Indicates if the network clip its gradients
     */
    static constexpr bool has_clip_gradients() noexcept {
        return desc::parameters::template contains<clip_gradients>();
    }

    /*!
     * \brief Returns the type of weight decay used during training
     */
    static constexpr decay_type decay() noexcept {
        return get_value_l_v<weight_decay<decay_type::NONE>, typename desc::parameters>;
    }
};

template <typename Network>
concept dynamic_network = dbn_traits<Network>::is_dynamic();

template <typename Network>
concept static_network = !dbn_traits<Network>::is_dynamic();

/** Functions to get the dimensions of network regardless of dynamic or not **/

/*!
 * \brief Return the network output size
 */
template <static_network Network>
constexpr size_t dbn_output_size(const Network& /*network*/) {
    return Network::output_size();
}

/*!
 * \brief Return the network output size
 */
template <dynamic_network Network>
size_t dbn_output_size(const Network& network) {
    return network.output_size();
}

/*!
 * \brief Return the network concatenated output size
 */
template <static_network Network>
constexpr size_t dbn_full_output_size(const Network& /*network*/) {
    return Network::full_output_size();
}

/*!
 * \brief Return the network concatenated output size
 */
template <dynamic_network Network>
size_t dbn_full_output_size(const Network& network) {
    return network.full_output_size();
}

/*!
 * \brief Return the network input size
 */
template <static_network Network>
constexpr size_t dbn_input_size(const Network& /*network*/) {
    return Network::input_size();
}

/*!
 * \brief Return the network input size
 */
template <dynamic_network Network>
size_t dbn_input_size(const Network& network) {
    return network.input_size();
}

template <typename Network, typename Layer>
struct transform_output_type {
    static constexpr auto dimensions = dbn_traits<Network>::is_convolutional() ? 4 : 2;

    using weight  = typename Network::weight; ///< The data type for this layer
    using type = etl::dyn_matrix<weight, dimensions>;
};

template <typename Network, typename Layer>
using transform_output_type_t = typename transform_output_type<Network, Layer>::type;

} //end of dll namespace
