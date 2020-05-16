//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "util/tmp.hpp"
#include "base_traits.hpp"
#include "layer_fwd.hpp"
#include "base_conf.hpp"

namespace dll {

/*!
 * \brief Type Traits to get information on layer type
 */
template <typename Layer>
struct layer_traits {
    using layer_t     = Layer;                      ///< The layer type being inspected
    using base_traits = layer_base_traits<layer_t>; ///< The base traits for the layer

    /*!
     * \brief Indicates if the layer is neural (dense or conv)
     */
    static constexpr bool is_neural_layer() {
        return base_traits::is_neural;
    }

    /*!
     * \brief Indicates if the layer is dense
     */
    static constexpr bool is_dense_layer() {
        return base_traits::is_dense;
    }

    /*!
     * \brief Indicates if the layer is convolutional
     */
    static constexpr bool is_convolutional_layer() {
        return base_traits::is_conv;
    }

    /*!
     * \brief Indicates if the layer is convolutional
     */
    static constexpr bool is_deconvolutional_layer() {
        return base_traits::is_deconv;
    }

    /*!
     * \brief Indicates if the layer is a standard (non-rbm) layer.
     */
    static constexpr bool is_standard_layer() {
        return base_traits::is_standard;
    }

    /*!
     * \brief Indicates if the layer is a standard (non-rbm) dense layer.
     */
    static constexpr bool is_standard_dense_layer() {
        return is_standard_layer() && is_dense_layer();
    }

    /*!
     * \brief Indicates if the layer is a standard (non-rbm) convolutionl layer.
     */
    static constexpr bool is_standard_convolutional_layer() {
        return is_standard_layer() && is_convolutional_layer();
    }

    /*!
     * \brief Indicates if the layer is a standard (non-rbm) deconvolutionl layer.
     */
    static constexpr bool is_standard_deconvolutional_layer() {
        return is_standard_layer() && is_deconvolutional_layer();
    }

    /*!
     * \brief Indicates if this layer is a RBM layer.
     */
    static constexpr bool is_rbm_layer() {
        return base_traits::is_rbm;
    }

    /*!
     * \brief Indicates if this layer is a dense RBM layer.
     */
    static constexpr bool is_dense_rbm_layer() {
        return is_rbm_layer() && is_dense_layer();
    }

    /*!
     * \brief Indicates if the layer is convolutional
     */
    static constexpr bool is_convolutional_rbm_layer() {
        return is_rbm_layer() && is_convolutional_layer();
    }

    /*!
     * \brief Indicates if this layer is a pooling layer.
     */
    static constexpr bool is_pooling_layer() {
        return base_traits::is_pooling;
    }

    /*!
     * \brief Indicates if this layer is a pooling layer.
     */
    static constexpr bool is_unpooling_layer() {
        return base_traits::is_unpooling;
    }

    /*!
     * \brief Indicates if this layer is a transformation layer.
     */
    static constexpr bool is_transform_layer() {
        return base_traits::is_transform;
    }

    /*!
     * \brief Indicates if this layer keeps the same type
     */
    static constexpr bool has_same_type() {
        return is_transform_layer();
    }

    /*!
     * \brief Indicates if this layer is trained or not.
     */
    static constexpr bool is_trained() {
        return is_neural_layer();
    }

    /*!
     * \brief Indicates if this layer is pretrained or not.
     */
    static constexpr bool is_pretrained() {
        return is_rbm_layer();
    }

    /*!
     * \brief Indicates if the layer is dynamic
     */
    static constexpr bool is_dynamic() {
        return base_traits::is_dynamic;
    }

    static constexpr bool pretrain_last() {
        return base_traits::pretrain_last;
    }
};

/*!
 * \brief Type Traits to get information on layer type
 */
template <typename Layer>
struct rbm_layer_traits {
    using layer_t     = Layer;                          ///< The layer type being inspected
    using base_traits = rbm_layer_base_traits<layer_t>; ///< The base traits for the layer

    /*!
     * \brief Indicates if the RBM must be trained with momentum or not
     */
    static constexpr bool has_momentum() {
        return base_traits::has_momentum;
    }

    static constexpr bool has_clip_gradients() {
        return base_traits::has_clip_gradients;
    }

    /*!
     * \brief Indicates if the RBM training is made verbose.
     */
    static constexpr bool is_verbose() {
        return base_traits::is_verbose;
    }

    /*!
     * \brief Indicates if the RBM must be trained with shuffle or not
     */
    static constexpr bool has_shuffle() {
        return base_traits::has_shuffle;
    }

    /*!
     * \brief Indicates if the RBM is only to be used inside a DBN.
     *
     * This can save some memory
     */
    static constexpr bool is_dbn_only() {
        return base_traits::is_dbn_only;
    }

    /*!
     * \brief Indicates if the RBM must be trained with sparsity or not
     */
    static constexpr bool has_sparsity() {
        return base_traits::has_sparsity;
    }

    static constexpr dll::sparsity_method sparsity_method() {
        return base_traits::sparsity_method;
    }

    /*!
     * \brief Return the bias mode t be used for the LEE sparsity regularization
     */
    static constexpr enum dll::bias_mode bias_mode() {
        return base_traits::bias_mode;
    }

    /*!
     * \brief Return the type of weight decay
     */
    static constexpr decay_type decay() {
        return base_traits::decay;
    }

    /*!
     * \brief Indicates if the weights of the RBM should be initialized using
     * the inputs according to Hinton
     */
    static constexpr bool init_weights() {
        return base_traits::has_init_weights;
    }

    /*!
     * \brief Indicates if the RBM's free energy is displayed while training.
     */
    static constexpr bool free_energy() {
        return base_traits::has_free_energy;
    }
};

template <typename T>
using decay_layer_traits = layer_traits<std::decay_t<T>>;

/*!
 * \brief Return the number of input channels of the given CRBM
 */
template <typename RBM, cpp_enable_iff(layer_traits<RBM>::is_dynamic())>
size_t get_nc(const RBM& rbm) {
    return rbm.nc;
}

/*!
 * \brief Return the number of input channels of the given CRBM
 */
template <typename RBM, cpp_disable_iff(layer_traits<RBM>::is_dynamic())>
constexpr size_t get_nc(const RBM&) {
    return RBM::NC;
}

/*!
 * \brief Return the number of filters of the given CRBM
 */
template <typename RBM, cpp_enable_iff(layer_traits<RBM>::is_dynamic())>
size_t get_k(const RBM& rbm) {
    return rbm.k;
}

/*!
 * \brief Return the number of filters of the given CRBM
 */
template <typename RBM, cpp_disable_iff(layer_traits<RBM>::is_dynamic())>
constexpr size_t get_k(const RBM&) {
    return RBM::K;
}

/*!
 * \brief Return the first dimension of the inputs
 */
template <typename RBM, cpp_enable_iff(layer_traits<RBM>::is_dynamic())>
size_t get_nv1(const RBM& rbm) {
    return rbm.nv1;
}

/*!
 * \brief Return the first dimension of the inputs
 */
template <typename RBM, cpp_disable_iff(layer_traits<RBM>::is_dynamic())>
constexpr size_t get_nv1(const RBM&) {
    return RBM::NV1;
}

/*!
 * \brief Return the second dimension of the inputs
 */
template <typename RBM, cpp_enable_iff(layer_traits<RBM>::is_dynamic())>
size_t get_nv2(const RBM& rbm) {
    return rbm.nv2;
}

/*!
 * \brief Return the second dimension of the inputs
 */
template <typename RBM, cpp_disable_iff(layer_traits<RBM>::is_dynamic())>
constexpr size_t get_nv2(const RBM&) {
    return RBM::NV2;
}

/*!
 * \brief Return the first dimension of the filters
 */
template <typename RBM, cpp_enable_iff(layer_traits<RBM>::is_dynamic())>
size_t get_nw1(const RBM& rbm) {
    return rbm.nw1;
}

/*!
 * \brief Return the first dimension of the filters
 */
template <typename RBM, cpp_disable_iff(layer_traits<RBM>::is_dynamic())>
constexpr size_t get_nw1(const RBM&) {
    return RBM::NW1;
}

/*!
 * \brief Return the second dimension of the filters
 */
template <typename RBM, cpp_enable_iff(layer_traits<RBM>::is_dynamic())>
size_t get_nw2(const RBM& rbm) {
    return rbm.nw2;
}

/*!
 * \brief Return the second dimension of the filters
 */
template <typename RBM, cpp_disable_iff(layer_traits<RBM>::is_dynamic())>
constexpr size_t get_nw2(const RBM&) {
    return RBM::NW2;
}

/*!
 * \brief Return the number of visible units of the given RBM
 * \param rbm The RBM to get the information from
 */
template <typename RBM, cpp_enable_iff(layer_traits<RBM>::is_dynamic())>
size_t num_visible(const RBM& rbm) {
    return rbm.num_visible;
}

/*!
 * \brief Return the number of visible units of the given RBM
 * \param rbm The RBM to get the information from
 */
template <typename RBM, cpp_disable_iff(layer_traits<RBM>::is_dynamic())>
constexpr size_t num_visible(const RBM&) {
    return RBM::desc::num_visible;
}

/*!
 * \brief Return the number of hidden units of the given RBM
 * \param rbm The RBM to get the information from
 */
template <typename RBM, cpp_enable_iff(layer_traits<RBM>::is_dynamic())>
size_t num_hidden(const RBM& rbm) {
    return rbm.num_hidden;
}

/*!
 * \brief Return the number of hidden units of the given RBM
 * \param rbm The RBM to get the information from
 */
template <typename RBM, cpp_disable_iff(layer_traits<RBM>::is_dynamic())>
constexpr size_t num_hidden(const RBM&) {
    return RBM::desc::num_hidden;
}

/*!
 * \brief Return the output size of the given RBM
 * \param rbm The RBM to get the information from
 */
template <typename RBM, cpp_disable_iff(layer_traits<RBM>::is_dynamic())>
constexpr size_t output_size(const RBM&) {
    return RBM::output_size();
}

/*!
 * \brief Return the output size of the given RBM
 * \param rbm The RBM to get the information from
 */
template <typename RBM, cpp_enable_iff(layer_traits<RBM>::is_dynamic())>
size_t output_size(const RBM& rbm) {
    return rbm.output_size();
}

/*!
 * \brief Return the input size of the given RBM
 * \param rbm The RBM to get the information from
 */
template <typename RBM, cpp_enable_iff(layer_traits<RBM>::is_dynamic())>
size_t input_size(const RBM& rbm) {
    return rbm.input_size();
}

/*!
 * \brief Return the input size of the given RBM
 * \param rbm The RBM to get the information from
 */
template <typename RBM, cpp_disable_iff(layer_traits<RBM>::is_dynamic())>
constexpr size_t input_size(const RBM&) {
    return RBM::input_size();
}

} //end of dll namespace
