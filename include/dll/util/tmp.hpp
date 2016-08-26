//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/tmp.hpp"

namespace dll {

namespace detail {

template <typename V, typename... Args>
struct is_valid;

template <typename V, typename T1, typename... Args>
struct is_valid<V, T1, Args...> : cpp::bool_constant_c<cpp::and_u<V::template contains<typename T1::type_id>(), is_valid<V, Args...>::value>> {};

template <typename V>
struct is_valid<V> : std::true_type {};

template <typename D, typename... Args>
struct get_value;

template <typename D, typename T2, typename... Args>
struct get_value<D, T2, Args...> : cpp::conditional_constant<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_value<D, Args...>> {};

template <typename D>
struct get_value<D> : cpp::auto_constant<D> {};

template <typename D, typename L>
struct get_value_l;

template <typename D, typename... T>
struct get_value_l<D, cpp::type_list<T...>> : cpp::auto_constant<get_value<D, T...>> {};

template <typename D, typename... Args>
struct get_type;

//Simply using conditional_t is not enough since T2::value can be a real value and not a type and therefore should not always be evaluated
template <typename D, typename T2, typename... Args>
struct get_type<D, T2, Args...> {
    using value = typename cpp::conditional_type_constant_c<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_type<D, Args...>>::type;
};

template <typename D>
struct get_type<D> {
    using value = typename D::value;
};

template <typename D, typename... Args>
struct get_template_type;

template <typename D, typename T2, typename... Args>
struct get_template_type<D, T2, Args...> {
    template <typename RBM>
    using value = typename cpp::conditional_template_type_constant_c<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_template_type<D, Args...>>::template type<RBM>;
};

template <typename D>
struct get_template_type<D> {
    template <typename RBM>
    using value = typename D::template value<RBM>;
};

template <bool C, typename V1, typename V2>
struct conditional_template_type_tb_constant_c;

template <typename V1, typename V2>
struct conditional_template_type_tb_constant_c<true, V1, V2> {
    template <typename T, bool C>
    using type = typename V1::template value<T, C>;
};

template <typename V1, typename V2>
struct conditional_template_type_tb_constant_c<false, V1, V2> {
    template <typename T, bool C>
    using type = typename V2::template value<T, C>;
};

template <typename D, typename... Args>
struct get_template_type_tb;

template <typename D, typename T2, typename... Args>
struct get_template_type_tb<D, T2, Args...> {
    template <typename RBM, bool Denoising>
    using value = typename conditional_template_type_tb_constant_c<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_template_type_tb<D, Args...>>::template type<RBM, Denoising>;
};

template <typename D>
struct get_template_type_tb<D> {
    template <typename RBM, bool Denoising>
    using value = typename D::template value<RBM, Denoising>;
};

} //end of namespace detail

template <typename Tuple, typename Functor, std::size_t I1>
void for_each_type_sub(Functor&& functor, const std::index_sequence<I1>& /* i */) {
    functor(static_cast<std::tuple_element_t<I1, Tuple>*>(nullptr));
}

template <typename Tuple, typename Functor, std::size_t I1, std::size_t... I, cpp_enable_if((sizeof...(I) > 0))>
void for_each_type_sub(Functor&& functor, const std::index_sequence<I1, I...>& /* i */) {
    functor(static_cast<std::tuple_element_t<I1, Tuple>*>(nullptr));
    for_each_type_sub<Tuple>(functor, std::index_sequence<I...>());
}

template <typename Tuple, typename Functor>
void for_each_type(Functor&& functor) {
    for_each_type_sub<Tuple>(functor, std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

//This allows to create a fast matrix type with an effective size of zero
//when it is not used, although this fast matrix is viewed as having
//the correct size

template <bool C, typename W, std::size_t... Dims>
struct conditional_fast_matrix {
    using type = std::conditional_t<
        C,
        etl::fast_matrix<W, Dims...>,
        etl::fast_matrix_impl<W, std::array<W, 0>, etl::order::RowMajor, Dims...>>;
};

template <bool C, typename W, std::size_t... Dims>
using conditional_fast_matrix_t = typename conditional_fast_matrix<C, W, Dims...>::type;

/* Utilities to build dyn layer types from normal layer types */

template <template <typename> class Layer, template<typename...> class Desc, typename Index, typename... Args>
struct build_dyn_layer_t ;

template <template <typename> class Layer, template<typename...> class Desc, size_t... I, typename... Args>
struct build_dyn_layer_t <Layer, Desc, std::index_sequence<I...>, Args...> {
    using type = Layer<Desc<cpp::nth_type_t<I, Args...>...>>;
};

template <typename Index, size_t N>
struct sequence_add;

template <size_t... I, size_t N>
struct sequence_add <std::index_sequence<I...>, N> {
    using type = std::index_sequence<I..., N>;
};

template <typename T, size_t F, size_t L, typename Acc, typename... Args>
struct remove_type_id_impl {
    using new_acc = std::conditional_t<
        std::is_same<typename cpp::nth_type_t<F, Args...>::type_id, T>::value,
        Acc,
        typename sequence_add<Acc, F>::type
        >;

    using type = typename remove_type_id_impl<T, F+1, L, new_acc, Args...>::type;
};

template <typename T, size_t F, typename Acc, typename... Args>
struct remove_type_id_impl <T, F, F, Acc, Args...> {
    using type = Acc;
};

// TODO For now, this cannot be chained
template <typename T, typename... Args>
using remove_type_id = typename remove_type_id_impl<T, 0, sizeof...(Args), std::index_sequence<>, Args...>::type;

} //end of dll namespace
