//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_LAYER_TRAITS_HPP
#define DLL_LAYER_TRAITS_HPP

#include "tmp.hpp"
#include "layer_fwd.hpp"
#include "base_conf.hpp"

namespace dll {

/*!
 * \brief Type Traits to get information on layer type
 */
template <typename Layer>
struct layer_traits {
    using layer_t = Layer;

    /*!
     * \brief Indicates if the layer is a standard layer.
     */
    static constexpr bool is_standard_layer() {
        return is_dense_layer() || is_convolutional_layer();
    }

    /*!
     * \brief Indicates if the layer is a standard dense layer.
     */
    static constexpr bool is_dense_layer() {
        return cpp::is_specialization_of<dense_layer, layer_t>::value;
    }

    /*!
     * \brief Indicates if the layer is a standard convolutionl layer.
     */
    static constexpr bool is_convolutional_layer() {
        return cpp::is_specialization_of<conv_layer, layer_t>::value;
    }

    /*!
     * \brief Indicates if this layer is a RBM layer.
     */
    static constexpr bool is_rbm_layer() {
        return is_standard_rbm_layer() || is_convolutional_rbm_layer();
    }

    /*!
     * \brief Indicates if this layer is a standard (non-convolutional) RBM layer.
     */
    static constexpr bool is_standard_rbm_layer() {
        return cpp::is_specialization_of<dyn_rbm, layer_t>::value || cpp::is_specialization_of<rbm, layer_t>::value;
    }

    /*!
     * \brief Indicates if the layer is convolutional
     */
    static constexpr bool is_convolutional_rbm_layer() {
        return cpp::is_specialization_of<conv_rbm, layer_t>::value || cpp::is_specialization_of<conv_rbm_mp, layer_t>::value;
    }

    /*!
     * \brief Indicates if this layer is a pooling layer.
     */
    static constexpr bool is_pooling_layer() {
        return cpp::is_specialization_of<mp_layer_3d, layer_t>::value || cpp::is_specialization_of<avgp_layer_3d, layer_t>::value;
    }

    /*!
     * \brief Indicates if this layer is a max pooling layer.
     */
    static constexpr bool is_max_pooling_layer() {
        return cpp::is_specialization_of<mp_layer_3d, layer_t>::value;
    }

    /*!
     * \brief Indicates if this layer is a avg pooling layer.
     */
    static constexpr bool is_avg_pooling_layer() {
        return cpp::is_specialization_of<avgp_layer_3d, layer_t>::value;
    }

    /*!
     * \brief Indicates if this layer is a transformation layer.
     */
    static constexpr bool is_transform_layer() {
        return cpp::is_specialization_of<binarize_layer, layer_t>::value || cpp::is_specialization_of<normalize_layer, layer_t>::value || cpp::is_specialization_of<scale_layer, layer_t>::value;
    }

    /*!
     * \brief Indicates if this layer is a transformation layer.
     */
    static constexpr bool is_multiplex_layer() {
        return cpp::is_specialization_of<patches_layer, layer_t>::value || cpp::is_specialization_of<patches_layer_padh, layer_t>::value;
    }

    /*!
     * \brief Indicates if this layer is trained or not.
     */
    static constexpr bool is_trained() {
        return !is_transform_layer() && !is_multiplex_layer() && !is_pooling_layer();
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
        return cpp::is_specialization_of<dyn_rbm, layer_t>::value;
    }

    /*!
     * \brief Indicates if the layer is convolutional and has probabilistic max
     * pooling
     */
    static constexpr bool has_probabilistic_max_pooling() {
        return cpp::is_specialization_of<conv_rbm_mp, layer_t>::value;
    }

    /*!
     * \brief Indicates if this layer should be trained if it is the last layer.
     */
    template <cpp_enable_if_cst(layer_traits<layer_t>::is_rbm_layer())>
    static constexpr bool pretrain_last() {
        //Softmax unit should not be pretrained
        return layer_t::hidden_unit != unit_type::SOFTMAX;
    }

    template <cpp_disable_if_cst(layer_traits<layer_t>::is_rbm_layer())>
    static constexpr bool pretrain_last() {
        //if the pooling layer is the last, we spare the time to activate the previous layer by not training it
        //since training pooling layer is a nop, that doesn't change anything
        return false;
    }

    static constexpr std::size_t input_size() {
        return layer_t::input_size();
    }

    static constexpr std::size_t output_size() {
        return layer_t::output_size();
    }

    static constexpr std::size_t batch_size() {
        return detail::get_value_l<dll::batch_size<1>, typename layer_t::desc::parameters>::value;
    }

    static constexpr bool has_momentum() {
        return layer_t::desc::parameters::template contains<momentum>();
    }

    static constexpr bool is_parallel_mode() {
        return layer_t::desc::parameters::template contains<parallel_mode>();
    }

    static constexpr bool is_serial() {
        return layer_t::desc::parameters::template contains<serial>();
    }

    static constexpr bool is_verbose() {
        return layer_t::desc::parameters::template contains<verbose>();
    }

    static constexpr bool has_shuffle() {
        return layer_t::desc::parameters::template contains<shuffle>();
    }

    static constexpr bool is_dbn_only() {
        return layer_t::desc::parameters::template contains<dbn_only>();
    }

    static constexpr bool is_memory() {
        return layer_t::desc::parameters::template contains<memory>();
    }

    static constexpr bool has_sparsity() {
        return sparsity_method() != dll::sparsity_method::NONE;
    }

    static constexpr dll::sparsity_method sparsity_method() {
        return detail::get_value_l<sparsity<dll::sparsity_method::NONE>, typename layer_t::desc::parameters>::value;
    }

    static constexpr enum dll::bias_mode bias_mode() {
        return detail::get_value_l<bias<dll::bias_mode::SIMPLE>, typename layer_t::desc::parameters>::value;
    }

    static constexpr decay_type decay() {
        return detail::get_value_l<weight_decay<decay_type::NONE>, typename layer_t::desc::parameters>::value;
    }

    static constexpr bool init_weights() {
        return layer_t::desc::parameters::template contains<dll::init_weights>();
    }

    static constexpr bool free_energy() {
        return layer_t::desc::parameters::template contains<dll::free_energy>();
    }
};

template <typename T>
using decay_layer_traits = layer_traits<std::decay_t<T>>;

template <typename RBM, cpp_enable_if(layer_traits<RBM>::is_dynamic())>
std::size_t get_batch_size(const RBM& rbm) {
    return rbm.batch_size;
}

template <typename RBM, cpp_disable_if(layer_traits<RBM>::is_dynamic())>
constexpr std::size_t get_batch_size(const RBM&) {
    return layer_traits<RBM>::batch_size();
}

template <typename RBM, cpp_enable_if(layer_traits<RBM>::is_dynamic())>
std::size_t num_visible(const RBM& rbm) {
    return rbm.num_visible;
}

template <typename RBM, cpp_disable_if(layer_traits<RBM>::is_dynamic())>
constexpr std::size_t num_visible(const RBM&) {
    return RBM::desc::num_visible;
}

template <typename RBM, cpp_enable_if(layer_traits<RBM>::is_dynamic())>
std::size_t num_hidden(const RBM& rbm) {
    return rbm.num_hidden;
}

template <typename RBM, cpp_disable_if(layer_traits<RBM>::is_dynamic())>
constexpr std::size_t num_hidden(const RBM&) {
    return RBM::desc::num_hidden;
}

template <typename RBM, cpp_disable_if(layer_traits<RBM>::is_dynamic())>
constexpr std::size_t output_size(const RBM&) {
    return layer_traits<RBM>::output_size();
}

template <typename RBM, cpp_enable_if(layer_traits<RBM>::is_dynamic())>
std::size_t output_size(const RBM& rbm) {
    return rbm.num_hidden;
}

template <typename RBM, cpp_enable_if(layer_traits<RBM>::is_dynamic())>
std::size_t input_size(const RBM& rbm) {
    return rbm.num_visible;
}

template <typename RBM, cpp_disable_if(layer_traits<RBM>::is_dynamic())>
constexpr std::size_t input_size(const RBM&) {
    return layer_traits<RBM>::input_size();
}

//TODO This should probably be moved into the traits class

template <typename Layer>
struct is_dense : cpp::bool_constant<decay_layer_traits<Layer>::is_dense_layer() || decay_layer_traits<Layer>::is_standard_rbm_layer()> {};

template <typename Layer>
struct is_conv : cpp::bool_constant<decay_layer_traits<Layer>::is_convolutional_layer() || decay_layer_traits<Layer>::is_convolutional_rbm_layer()> {};

template <typename Layer>
struct is_neural : cpp::or_c<is_dense<Layer>, is_conv<Layer>> {};

} //end of dll namespace

#endif
