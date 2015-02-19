//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_RBM_TRAITS_HPP
#define DLL_RBM_TRAITS_HPP

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

/*!
 * \brief Type Traits to get information on RBM type
 */
template<typename RBM>
struct rbm_traits {
    using rbm_t = RBM;
    using desc = typename rbm_t::desc;

    HAS_STATIC_FIELD(BatchSize, has_batch_size_field)
    HAS_STATIC_FIELD(Bias, has_bias_field)

    /*!
     * \brief Indicates if the RBM is convolutional
     */
    static constexpr bool is_convolutional(){
        return cpp::is_specialization_of<conv_rbm, rbm_t>::value
            || cpp::is_specialization_of<conv_rbm_mp, rbm_t>::value;
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

    template<cpp_enable_if_cst(has_batch_size_field<desc>::value)>
    static constexpr std::size_t batch_size(){
        return rbm_t::desc::BatchSize;
    }

    template<cpp_disable_if_cst(has_batch_size_field<desc>::value)>
    static constexpr std::size_t batch_size(){
        return 1;
    }

    static constexpr bool has_momentum(){
        return desc::parameters::template contains<momentum>();
    }

    static constexpr bool is_parallel(){
        return desc::parameters::template contains<parallel>();
    }

    static constexpr bool is_verbose(){
        return desc::parameters::template contains<verbose>();
    }

    static constexpr bool has_shuffle(){
        return desc::parameters::template contains<shuffle>();
    }

    static constexpr bool has_sparsity(){
        return sparsity_method() != dll::sparsity_method::NONE;
    }

    static constexpr dll::sparsity_method sparsity_method(){
        return detail::get_value_l<sparsity<dll::sparsity_method::NONE>, typename desc::parameters>::value;
    }

    template<cpp_enable_if_cst(has_bias_field<desc>::value)>
    static constexpr enum bias_mode bias_mode(){
        return rbm_t::desc::Bias;
    }

    template<cpp_disable_if_cst(has_bias_field<desc>::value)>
    static constexpr enum bias_mode bias_mode(){
        return dll::bias_mode::SIMPLE;
    }

    static constexpr decay_type decay(){
        return detail::get_value_l<weight_decay<decay_type::NONE>, typename desc::parameters>::value;
    }

    static constexpr bool init_weights(){
        return desc::parameters::template contains<dll::init_weights>();
    }

    static constexpr bool free_energy(){
        return desc::parameters::template contains<dll::free_energy>();
    }
};

template<typename RBM, cpp::enable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
std::size_t get_batch_size(const RBM& rbm){
    return rbm.batch_size;
}

template<typename RBM, cpp::disable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t get_batch_size(const RBM&){
    return rbm_traits<RBM>::batch_size();
}

template<typename RBM, cpp::enable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
std::size_t num_visible(const RBM& rbm){
    return rbm.num_visible;
}

template<typename RBM, cpp::disable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t num_visible(const RBM&){
    return RBM::desc::num_visible;
}

template<typename RBM, cpp::enable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
std::size_t num_hidden(const RBM& rbm){
    return rbm.num_hidden;
}

template<typename RBM, cpp::disable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t num_hidden(const RBM&){
    return RBM::desc::num_hidden;
}

template<typename RBM, cpp::disable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t output_size(const RBM&){
    return rbm_traits<RBM>::output_size();
}

template<typename RBM, cpp::enable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
std::size_t output_size(const RBM& rbm){
    return rbm.num_hidden;
}

template<typename RBM, cpp::enable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
std::size_t input_size(const RBM& rbm){
    return rbm.num_visible;
}

template<typename RBM, cpp::disable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t input_size(const RBM&){
    return rbm_traits<RBM>::input_size();
}

} //end of dll namespace

#endif
