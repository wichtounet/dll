//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_TMP_HPP
#define DLL_TMP_HPP

#include "cpp_utils/tmp.hpp"

namespace dll {

namespace detail {

template<typename T1, typename... Args>
struct is_present;

template<typename T1, typename T2, typename... Args>
struct is_present<T1, T2, Args...> : cpp::bool_constant_c<cpp::or_c<std::is_same<T1, T2>, is_present<T1, Args...>>> {};

template<typename T1>
struct is_present<T1> : std::false_type {};

template<typename T1, typename T2>
struct is_in_list ;

template<typename T1, typename... T>
struct is_in_list<T1, cpp::type_list<T...>> : cpp::bool_constant_c<is_present<T1, T...>> {} ;

template<typename... Valid>
struct tmp_list {
    template<typename T>
    struct check : cpp::bool_constant_c<is_present<typename T::type_id, Valid...>> {};
};

template<typename V, typename... Args>
struct is_valid;

template<typename V, typename T1, typename... Args>
struct is_valid <V, T1, Args...> : cpp::bool_constant_c<cpp::and_c<typename V::template check<T1>, is_valid<V, Args...>>> {};

template<typename V>
struct is_valid <V> : std::true_type {};

template<typename D, typename... Args>
struct get_value;

template<typename D, typename T2, typename... Args>
struct get_value<D, T2, Args...> : cpp::conditional_constant<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_value<D, Args...>> {};

template<typename D>
struct get_value<D> : cpp::auto_constant<D> {};

template<typename D, typename... Args>
struct get_type;

//Simply using conditional_t is not enough since T2::value can be a real value and not a type and therefore should not always be evaluated
template<typename D, typename T2, typename... Args>
struct get_type<D, T2, Args...> {
    using value = typename cpp::conditional_type_constant_c<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_type<D, Args...>>::type;
};

template<typename D>
struct get_type<D> {
    using value = typename D::value;
};

template<typename D, typename... Args>
struct get_template_type;

template<typename D, typename T2, typename... Args>
struct get_template_type<D, T2, Args...> {
    template<typename RBM>
    using value = typename cpp::conditional_template_type_constant_c<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_template_type<D, Args...>>::template type<RBM>;
};

template<typename D>
struct get_template_type<D> {
    template<typename RBM>
    using value = typename D::template value<RBM>;
};

} //end of namespace detail

template<typename Tuple, typename Functor, std::size_t I1>
void for_each_type_sub(Functor&& functor, const std::index_sequence<I1>& /* i */){
    functor(static_cast<std::tuple_element_t<I1, Tuple>*>(nullptr));
}

template<typename Tuple, typename Functor, std::size_t I1, std::size_t... I, cpp::enable_if_u<(sizeof...(I) > 0)> = cpp::detail::dummy>
void for_each_type_sub(Functor&& functor, const std::index_sequence<I1, I...>& /* i */){
    functor(static_cast<std::tuple_element_t<I1, Tuple>*>(nullptr));
    for_each_type_sub<Tuple>(functor, std::index_sequence<I...>());
}

template<typename Tuple, typename Functor>
void for_each_type(Functor&& functor){
    for_each_type_sub<Tuple>(functor, std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

} //end of dll namespace

#endif
