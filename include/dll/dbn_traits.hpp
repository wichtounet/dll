//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "util/tmp.hpp"
#include "decay_type.hpp"

namespace dll {

/*!
 * \brief Type Traits to get information on DBN type
 */
template <typename DBN>
struct dbn_traits {
    using dbn_t = DBN;
    using desc  = typename dbn_t::desc; ///< The descriptor of the layer

    /*!
     * \brief Indicates if the DBN is convolutional
     */
    static constexpr bool is_convolutional() noexcept {
        return desc::layers::is_convolutional;
    }

    /*!
     * \brief Indicates if the DBN is dynamic
     */
    static constexpr bool is_dynamic() noexcept {
        return desc::layers::is_dynamic;
    }

    /*!
     * \brief Get the updater type of the DBN.
     */
    static constexpr updater_type updater() noexcept {
        return desc::Updater;
    }

    /*!
     * \brief Indicates if the DBN runs in batch mode
     */
    static constexpr bool batch_mode() noexcept {
        return desc::parameters::template contains<dll::batch_mode>();
    }

    /*!
     * \brief Indicates if the DBN computes error on epoch.
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
     * \brief Indicates if the DBN shuffles the inputs before each
     * fine-tuning epoch.
     */
    static constexpr bool shuffle() noexcept {
        return desc::parameters::template contains<dll::shuffle>();
    }

    /*!
     * \brief Indicates if the DBN shuffles the inputs before
     * each pretraining epoch in batch mode.
     */
    static constexpr bool shuffle_pretrain() noexcept {
        return desc::parameters::template contains<dll::shuffle_pre>();
    }

    /*!
     * \brief Indicates if the DBN features are concatenated from all levels
     */
    static constexpr bool concatenate() noexcept {
        return desc::parameters::template contains<svm_concatenate>();
    }

    /*!
     * \brief Indicates if the DBN cannot use threading
     */
    static constexpr bool is_serial() noexcept {
        return desc::parameters::template contains<serial>();
    }

    /*!
     * \brief Indicates if the DBN is verbose
     */
    static constexpr bool is_verbose() noexcept {
        return desc::parameters::template contains<verbose>();
    }

    /*!
     * \brief Indicates if the DBN is verbose
     */
    static constexpr bool should_display_batch() noexcept {
        return !desc::parameters::template contains<no_batch_display>();
    }

    /*!
     * \brief Indicates if the DBN scales its features before sending to SVM.
     */
    static constexpr bool scale() noexcept {
        return desc::parameters::template contains<svm_scale>();
    }

    /*!
     * \brief Indicates if the DBN clip its gradients
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

/** Functions to get the dimensions of DBN regardless of dynamic or not **/

/*!
 * \brief Return the DBN output size
 */
template <typename DBN, cpp_disable_iff(dbn_traits<DBN>::is_dynamic())>
constexpr size_t dbn_output_size(const DBN& /*dbn*/) {
    return DBN::output_size();
}

/*!
 * \brief Return the DBN output size
 */
template <typename DBN, cpp_enable_iff(dbn_traits<DBN>::is_dynamic())>
size_t dbn_output_size(const DBN& dbn) {
    return dbn.output_size();
}

/*!
 * \brief Return the DBN concatenated output size
 */
template <typename DBN, cpp_disable_iff(dbn_traits<DBN>::is_dynamic())>
constexpr size_t dbn_full_output_size(const DBN& /*dbn*/) {
    return DBN::full_output_size();
}

/*!
 * \brief Return the DBN concatenated output size
 */
template <typename DBN, cpp_enable_iff(dbn_traits<DBN>::is_dynamic())>
size_t dbn_full_output_size(const DBN& dbn) {
    return dbn.full_output_size();
}

/*!
 * \brief Return the DBN input size
 */
template <typename DBN, cpp_disable_iff(dbn_traits<DBN>::is_dynamic())>
constexpr size_t dbn_input_size(const DBN& /*dbn*/) {
    return DBN::input_size();
}

/*!
 * \brief Return the DBN input size
 */
template <typename DBN, cpp_enable_iff(dbn_traits<DBN>::is_dynamic())>
size_t dbn_input_size(const DBN& dbn) {
    return dbn.input_size();
}

template <typename DBN, typename Layer>
struct transform_output_type {
    static constexpr auto dimensions = dbn_traits<DBN>::is_convolutional() ? 4 : 2;

    using weight  = typename DBN::weight; ///< The data type for this layer
    using type = etl::dyn_matrix<weight, dimensions>;
};

template <typename DBN, typename Layer>
using transform_output_type_t = typename transform_output_type<DBN, Layer>::type;

} //end of dll namespace
