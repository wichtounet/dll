//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_TRAITS_HPP
#define DLL_DBN_TRAITS_HPP

#include "tmp.hpp"
#include "decay_type.hpp"

namespace dll {

template<typename Desc>
struct conv_dbn;

template<typename Desc>
struct dyn_dbn;

/*!
 * \brief Type Traits to get information on DBN type
 */
template<typename DBN>
struct dbn_traits {
    using dbn_t = DBN;

    HAS_STATIC_FIELD(Momentum, has_momentum_field)
    HAS_STATIC_FIELD(Concatenate, has_concatenate_field)
    HAS_STATIC_FIELD(Decay, has_decay_field)

    /*!
     * \brief Indicates if the DBN is convolutional
     */
    static constexpr bool is_convolutional(){
        return cpp::is_specialization_of<conv_dbn, dbn_t>::value;
    }

    /*!
     * \brief Indicates if the DBN is dynamic
     */
    static constexpr bool is_dynamic(){
        return cpp::is_specialization_of<dyn_dbn, dbn_t>::value;
    }

    template<typename D = DBN, cpp::enable_if_u<has_momentum_field<typename D::desc>::value> = cpp::detail::dummy>
    static constexpr bool has_momentum(){
        return dbn_t::desc::Momentum;
    }

    template<typename D = DBN, cpp::disable_if_u<has_momentum_field<typename D::desc>::value> = cpp::detail::dummy>
    static constexpr bool has_momentum(){
        return false;
    }

    template<typename D = DBN, cpp::enable_if_u<has_concatenate_field<typename D::desc>::value> = cpp::detail::dummy>
    static constexpr bool concatenate(){
        return dbn_t::desc::Concatenate;
    }

    template<typename D = DBN, cpp::disable_if_u<has_concatenate_field<typename D::desc>::value> = cpp::detail::dummy>
    static constexpr bool concatenate(){
        return false;
    }

    template<typename D = DBN, cpp::enable_if_u<has_decay_field<typename D::desc>::value> = cpp::detail::dummy>
    static constexpr decay_type decay(){
        return dbn_t::desc::Decay;
    }

    template<typename D = DBN, cpp::disable_if_u<has_decay_field<typename D::desc>::value> = cpp::detail::dummy>
    static constexpr decay_type decay(){
        return decay_type::NONE;
    }
};

} //end of dll namespace

#endif
