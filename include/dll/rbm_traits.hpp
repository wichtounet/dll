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

    HAS_STATIC_FIELD(BatchSize, has_batch_size_field)
    HAS_STATIC_FIELD(Sparsity, has_sparsity_field)
    HAS_STATIC_FIELD(Decay, has_decay_field)
    HAS_STATIC_FIELD(Init, has_init_field)
    HAS_STATIC_FIELD(Momentum, has_momentum_field)
    HAS_STATIC_FIELD(Parallel, has_parallel_field)
    HAS_STATIC_FIELD(Bias, has_bias_field)
    HAS_STATIC_FIELD(Shuffle, has_shuffle_field)
    HAS_STATIC_FIELD(Free_Energy, has_free_energy_field)

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

    template<typename R = RBM, cpp::enable_if_c<has_batch_size_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr std::size_t batch_size(){
        return rbm_t::desc::BatchSize;
    }

    template<typename R = RBM, cpp::disable_if_c<has_batch_size_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr std::size_t batch_size(){
        return 1;
    }

    template<typename R = RBM, cpp::enable_if_c<has_momentum_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool has_momentum(){
        return rbm_t::desc::Momentum;
    }

    template<typename R = RBM, cpp::disable_if_c<has_momentum_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool has_momentum(){
        return false;
    }

    template<typename R = RBM, cpp::enable_if_c<has_parallel_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool is_parallel(){
        return rbm_t::desc::Parallel;
    }

    template<typename R = RBM, cpp::disable_if_c<has_parallel_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool is_parallel(){
        return false;
    }

    template<typename R = RBM, cpp::enable_if_c<has_shuffle_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool has_shuffle(){
        return rbm_t::desc::Shuffle;
    }

    template<typename R = RBM, cpp::disable_if_c<has_shuffle_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool has_shuffle(){
        return false;
    }

    template<typename R = RBM, cpp::enable_if_c<has_sparsity_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool has_sparsity(){
        return rbm_t::desc::Sparsity != dll::sparsity_method::NONE;
    }

    template<typename R = RBM, cpp::disable_if_c<has_sparsity_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool has_sparsity(){
        return false;
    }

    template<typename R = RBM, cpp::enable_if_c<has_sparsity_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr enum sparsity_method sparsity_method(){
        return rbm_t::desc::Sparsity;
    }

    template<typename R = RBM, cpp::disable_if_c<has_sparsity_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr enum sparsity_method sparsity_method(){
        return dll::sparsity_method::NONE;
    }

    template<typename R = RBM, cpp::enable_if_c<has_bias_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr enum bias_mode bias_mode(){
        return rbm_t::desc::Bias;
    }

    template<typename R = RBM, cpp::disable_if_c<has_bias_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr enum bias_mode bias_mode(){
        return dll::bias_mode::SIMPLE;
    }

    template<typename R = RBM, cpp::enable_if_c<has_decay_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr decay_type decay(){
        return rbm_t::desc::Decay;
    }

    template<typename R = RBM, cpp::disable_if_c<has_decay_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr decay_type decay(){
        return dll::decay_type::NONE;
    }

    template<typename R = RBM, cpp::enable_if_c<has_init_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool init_weights(){
        return rbm_t::desc::Init;
    }

    template<typename R = RBM, cpp::disable_if_c<has_init_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool init_weights(){
        return false;
    }

    template<typename R = RBM, cpp::enable_if_c<has_free_energy_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool free_energy(){
        return rbm_t::desc::Free_Energy;
    }

    template<typename R = RBM, cpp::disable_if_c<has_free_energy_field<typename R::desc>> = cpp::detail::dummy>
    static constexpr bool free_energy(){
        return false;
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
