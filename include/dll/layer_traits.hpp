//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_LAYER_TRAITS_HPP
#define DBN_LAYER_TRAITS_HPP

#include "tmp.hpp"

namespace dll {

#define HAS_STATIC_FIELD(field, name) \
template <class T> \
class name { \
    template<typename U, typename = \
    typename std::enable_if<!std::is_member_pointer<decltype(&U::field)>::value>::type> \
    static std::true_type check(int); \
    template <typename> \
    static std::false_type check(...); \
    public: \
    static constexpr const bool value = decltype(check<T>(0))::value; \
};

template<typename RBM>
struct rbm_traits {
    using rbm_t = RBM;

    HAS_STATIC_FIELD(Sparsity, has_sparsity_field)

    template<typename R = RBM, enable_if_u<has_sparsity_field<R>::value> = detail::dummy>
    static constexpr bool has_sparsity(){
        return rbm_t::Sparsity;
    }

    template<typename R = RBM, disable_if_u<has_sparsity_field<R>::value> = detail::dummy>
    static constexpr bool has_sparsity(){
        return false;
    }
};

} //end of dbn namespace

#endif