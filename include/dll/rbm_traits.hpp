//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_rbm_traits_HPP
#define DBN_rbm_traits_HPP

#include "tmp.hpp"
#include "decay_type.hpp"
#include "conv_rbm.hpp"
#include "conv_rbm_mp.hpp"

namespace dll {

template<typename RBM>
struct rbm_traits {
    using rbm_t = RBM;

    HAS_STATIC_FIELD(BatchSize, has_batch_size_field)
    HAS_STATIC_FIELD(Sparsity, has_sparsity_field)
    HAS_STATIC_FIELD(Decay, has_decay_field)
    HAS_STATIC_FIELD(Debug, has_debug_field)
    HAS_STATIC_FIELD(Init, has_init_field)
    HAS_STATIC_FIELD(Momentum, has_momentum_field)

    static constexpr bool is_convolutional(){
        return is_instantiation_of<conv_rbm, rbm_t>::value
            || is_instantiation_of<conv_rbm_mp, rbm_t>::value;
    }

    template<typename R = RBM, enable_if_u<has_batch_size_field<R>::value> = detail::dummy>
    static constexpr std::size_t batch_size(){
        return rbm_t::BatchSize;
    }

    template<typename R = RBM, disable_if_u<has_batch_size_field<R>::value> = detail::dummy>
    static constexpr std::size_t batch_size(){
        return 1;
    }

    template<typename R = RBM, enable_if_u<has_momentum_field<typename R::layer>::value> = detail::dummy>
    static constexpr bool has_momentum(){
        return rbm_t::layer::Momentum;
    }

    template<typename R = RBM, disable_if_u<has_momentum_field<typename R::layer>::value> = detail::dummy>
    static constexpr bool has_momentum(){
        return false;
    }

    template<typename R = RBM, enable_if_u<has_sparsity_field<R>::value> = detail::dummy>
    static constexpr bool has_sparsity(){
        return rbm_t::Sparsity;
    }

    template<typename R = RBM, disable_if_u<has_sparsity_field<R>::value> = detail::dummy>
    static constexpr bool has_sparsity(){
        return false;
    }

    template<typename R = RBM, enable_if_u<has_debug_field<R>::value> = detail::dummy>
    static constexpr bool debug_mode(){
        return rbm_t::Debug;
    }

    template<typename R = RBM, disable_if_u<has_debug_field<R>::value> = detail::dummy>
    static constexpr bool debug_mode(){
        return false;
    }

    template<typename R = RBM, enable_if_u<has_decay_field<R>::value> = detail::dummy>
    static constexpr decay_type decay(){
        return rbm_t::Decay;
    }

    template<typename R = RBM, disable_if_u<has_decay_field<R>::value> = detail::dummy>
    static constexpr decay_type decay(){
        return decay_type::NONE;
    }

    template<typename R = RBM, enable_if_u<has_init_field<R>::value> = detail::dummy>
    static constexpr bool init_weights(){
        return rbm_t::Init;
    }

    template<typename R = RBM, disable_if_u<has_init_field<R>::value> = detail::dummy>
    static constexpr bool init_weights(){
        return false;
    }
};

} //end of dbn namespace

#endif