//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_TMP_HPP
#define DLL_TMP_HPP

#include "cpp_utils/tmp.hpp" //for enable_if/disable_if stuff

namespace dll {

namespace detail {

template<typename T1, typename... Args>
struct is_present;

template<typename T1, typename T2, typename... Args>
struct is_present<T1, T2, Args...> :
    std::integral_constant<bool, cpp::or_u<std::is_same<T1, T2>::value, is_present<T1, Args...>::value>::value> {};

template<typename T1>
struct is_present<T1> : std::false_type{};

template<typename... Valid>
struct tmp_list {
    template<typename T>
    struct check : std::integral_constant<bool, is_present<typename T::type_id, Valid...>::value> {};
};

template<typename V, typename... Args>
struct is_valid;

template<typename V, typename T1, typename... Args>
struct is_valid <V, T1, Args...> :
    std::integral_constant<bool, cpp::and_u<V::template check<T1>::value, is_valid<V, Args...>::value>::value> {};

template<typename V>
struct is_valid <V> : std::true_type {};

//Since default template argument are not supported in partial class specialization, the following
//three extractors are more complicated and need a second class to perform SFINAE

template<typename D, typename... Args>
struct get_value;

template<typename D, typename T2, typename... Args>
struct get_value<D, T2, Args...> {
    template<typename D2, typename T22, typename Enable = void>
    struct get_value_2 {
        static constexpr const auto value = get_value<D, Args...>::value;
    };

    template<typename D2, typename T22>
    struct get_value_2 <D2, T22, std::enable_if_t<std::is_same<typename D2::type_id, typename T22::type_id>::value>> {
        static constexpr const auto value = T22::value;
    };

    static constexpr const auto value = get_value_2<D, T2>::value;
};

template<typename D>
struct get_value<D> {
    static constexpr const auto value = D::value;
};

template<typename D, typename... Args>
struct get_type;

template<typename D, typename T2, typename... Args>
struct get_type<D, T2, Args...> {
    template<typename D2, typename T22, typename Enable = void>
    struct get_type_2 {
        using type = typename get_type<D, Args...>::type;
    };

    template<typename D2, typename T22>
    struct get_type_2 <D2, T22, std::enable_if_t<std::is_same<typename D2::type_id, typename T22::type_id>::value>> {
        using type = typename T22::value;
    };

    using type = typename get_type_2<D, T2>::type;
};

template<typename D>
struct get_type<D> {
    using type = typename D::value;
};

template<typename D, typename... Args>
struct get_template_type;

template<typename D, typename T2, typename... Args>
struct get_template_type<D, T2, Args...> {
    template<typename D2, typename T22, typename Enable = void>
    struct get_template_type_2 {
        template<typename RBM>
        using type = typename get_template_type<D, Args...>::template type<RBM>;
    };

    template<typename D2, typename T22>
    struct get_template_type_2 <D2, T22, std::enable_if_t<std::is_same<typename D2::type_id, typename T22::type_id>::value>> {
        template<typename RBM>
        using type = typename T22::template value<RBM>;
    };

    template<typename RBM>
    using type = typename get_template_type_2<D, T2>::template type<RBM>;
};

template<typename D>
struct get_template_type<D> {
    template<typename RBM>
    using type = typename D::template value<RBM>;
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

} //end of dbn namespace

#endif
