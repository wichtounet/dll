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

/*!
 * \brief Type Traits to get information on DBN type
 */
template<typename DBN>
struct dbn_traits {
    using dbn_t = DBN;

    HAS_STATIC_FIELD(Momentum, has_momentum_field)

    template<typename D = DBN, enable_if_u<has_momentum_field<typename D::desc>::value> = ::detail::dummy>
    static constexpr bool has_momentum(){
        return dbn_t::desc::Momentum;
    }

    template<typename D = DBN, disable_if_u<has_momentum_field<typename D::desc>::value> = ::detail::dummy>
    static constexpr bool has_momentum(){
        return false;
    }
};

} //end of dbn namespace

#endif