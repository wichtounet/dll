//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_LAYER_TRAITS_HPP
#define DLL_LAYER_TRAITS_HPP

#include "tmp.hpp"
#include "decay_type.hpp"
#include "sparsity_method.hpp"

namespace dll {

template<typename Desc>
struct dyn_rbm;

template<typename Desc>
struct conv_rbm;

template<typename Desc>
struct conv_rbm_mp;

template<typename Desc>
struct mp_layer_3d;

template<typename Desc>
struct avgp_layer_3d;

/*!
 * \brief Type Traits to get information on RBM type
 */
template<typename RBM>
struct layer_traits {
    using rbm_t = RBM;

    /*!
     * \brief Indicates if the RBM is convolutional
     */
    static constexpr bool is_convolutional(){
        return cpp::is_specialization_of<conv_rbm, rbm_t>::value
            || cpp::is_specialization_of<conv_rbm_mp, rbm_t>::value;
    }

    /*!
     * \brief Indicates if this layer is a RBM layer.
     */
    static constexpr bool is_rbm_layer(){
        return !is_pooling_layer();
    }

    /*!
     * \brief Indicates if this layer is a pooling layer.
     */
    static constexpr bool is_pooling_layer(){
        return cpp::is_specialization_of<mp_layer_3d, rbm_t>::value
            || cpp::is_specialization_of<avgp_layer_3d, rbm_t>::value;
    }

    template<cpp_enable_if_cst(layer_traits<rbm_t>::is_rbm_layer())>
    static constexpr bool pretrain_last(){
        //Softmax unit should not be pretrained
        return rbm_t::hidden_unit != unit_type::SOFTMAX;
    }

    template<cpp_disable_if_cst(layer_traits<rbm_t>::is_rbm_layer())>
    static constexpr bool pretrain_last(){
        //if the pooling layer is the last, we spare the time to activate the previous layer by not training it
        //since training pooling layer is a nop, that doesn't change anything
        return false;
    }

    /*!
     * \brief Indicates if the RBM is dynamic
     */
    static constexpr bool is_dynamic(){
        return cpp::is_specialization_of<dyn_rbm, rbm_t>::value;
    }

    /*!
     * \brief Indicates if the RBM is convolutional and has probabilistic max
     * pooling
     */
    static constexpr bool has_probabilistic_max_pooling(){
        return cpp::is_specialization_of<conv_rbm_mp, rbm_t>::value;
    }

    static constexpr std::size_t input_size(){
        return rbm_t::input_size();
    }

    static constexpr std::size_t output_size(){
        return rbm_t::output_size();
    }

    static constexpr std::size_t batch_size(){
        return detail::get_value_l<dll::batch_size<1>, typename rbm_t::desc::parameters>::value;
    }

    static constexpr bool has_momentum(){
        return rbm_t::desc::parameters::template contains<momentum>();
    }

    static constexpr bool is_parallel(){
        return rbm_t::desc::parameters::template contains<parallel>();
    }

    static constexpr bool is_verbose(){
        return rbm_t::desc::parameters::template contains<verbose>();
    }

    static constexpr bool has_shuffle(){
        return rbm_t::desc::parameters::template contains<shuffle>();
    }

    static constexpr bool is_dbn_only(){
        return rbm_t::desc::parameters::template contains<dll::dbn_only>();
    }

    static constexpr bool has_sparsity(){
        return sparsity_method() != dll::sparsity_method::NONE;
    }

    static constexpr dll::sparsity_method sparsity_method(){
        return detail::get_value_l<sparsity<dll::sparsity_method::NONE>, typename rbm_t::desc::parameters>::value;
    }

    static constexpr enum dll::bias_mode bias_mode(){
        return detail::get_value_l<bias<dll::bias_mode::SIMPLE>, typename rbm_t::desc::parameters>::value;
    }

    static constexpr decay_type decay(){
        return detail::get_value_l<weight_decay<decay_type::NONE>, typename rbm_t::desc::parameters>::value;
    }

    static constexpr bool init_weights(){
        return rbm_t::desc::parameters::template contains<dll::init_weights>();
    }

    static constexpr bool free_energy(){
        return rbm_t::desc::parameters::template contains<dll::free_energy>();
    }
};

template<typename RBM, cpp::enable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
std::size_t get_batch_size(const RBM& rbm){
    return rbm.batch_size;
}

template<typename RBM, cpp::disable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t get_batch_size(const RBM&){
    return layer_traits<RBM>::batch_size();
}

template<typename RBM, cpp::enable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
std::size_t num_visible(const RBM& rbm){
    return rbm.num_visible;
}

template<typename RBM, cpp::disable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t num_visible(const RBM&){
    return RBM::desc::num_visible;
}

template<typename RBM, cpp::enable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
std::size_t num_hidden(const RBM& rbm){
    return rbm.num_hidden;
}

template<typename RBM, cpp::disable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t num_hidden(const RBM&){
    return RBM::desc::num_hidden;
}

template<typename RBM, cpp::disable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t output_size(const RBM&){
    return layer_traits<RBM>::output_size();
}

template<typename RBM, cpp::enable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
std::size_t output_size(const RBM& rbm){
    return rbm.num_hidden;
}

template<typename RBM, cpp::enable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
std::size_t input_size(const RBM& rbm){
    return rbm.num_visible;
}

template<typename RBM, cpp::disable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t input_size(const RBM&){
    return layer_traits<RBM>::input_size();
}

} //end of dll namespace

#endif
