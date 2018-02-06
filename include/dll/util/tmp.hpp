//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/etl.hpp"

namespace dll {

namespace detail {

/*!
 * \brief A conditional constant extracting the member value_1 from V1 or value V2 depending on the condition
 * \tparam C The boolean value
 * \tparam V1 The first value class
 * \tparam V2 The second value class
 */
template <bool C, typename V1, typename V2>
struct conditional_constant_v1;

/*!
 * \copydoc conditional_constant_v1
 */
template <typename V1, typename V2>
struct conditional_constant_v1<true, V1, V2> : std::integral_constant<decltype(V1::value_1), V1::value_1> {};

/*!
 * \copydoc conditional_constant_v1
 */
template <typename V1, typename V2>
struct conditional_constant_v1<false, V1, V2> : cpp::auto_constant<V2> {};

/*!
 * \brief A conditional constant extracting the member value_2 from V1 or value V2 depending on the condition
 * \tparam C The boolean value
 * \tparam V1 The first value class
 * \tparam V2 The second value class
 */
template <bool C, typename V1, typename V2>
struct conditional_constant_v2;

/*!
 * \copydoc conditional_constant_v2
 */
template <typename V1, typename V2>
struct conditional_constant_v2<true, V1, V2> : std::integral_constant<decltype(V1::value_2), V1::value_2> {};

/*!
 * \copydoc conditional_constant_v2
 */
template <typename V1, typename V2>
struct conditional_constant_v2<false, V1, V2> : cpp::auto_constant<V2> {};

template <typename V, typename... Args>
struct is_valid;

template <typename V, typename T1, typename... Args>
struct is_valid<V, T1, Args...> : cpp::bool_constant_c<cpp::and_u<V::template contains<typename T1::type_id>(), is_valid<V, Args...>::value>> {};

template <typename V>
struct is_valid<V> : std::true_type {};

template <typename V, typename... Args>
constexpr const bool is_valid_v = is_valid<V, Args...>::value;

/*!
 * \brief Extract the value corresponding to the given configuration element from the parameters.
 * \tparam D The configuration element type
 * \tparam Args The arguments to extract the value from
 */
template <typename D, typename... Args>
struct get_value;

/*!
 * \copydoc get_value
 */
template <typename D, typename T2, typename... Args>
struct get_value<D, T2, Args...> : cpp::conditional_constant<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_value<D, Args...>> {};

/*!
 * \copydoc get_value
 */
template <typename D>
struct get_value<D> : cpp::auto_constant<D> {};

/*!
 * \brief Extract the value corresponding to the given configuration element from the parameters.
 * \tparam D The configuration element type
 * \tparam Args The arguments to extract the value from
 */
template <typename D, typename... Args>
constexpr const auto get_value_v = get_value<D, Args...>::value;

/*!
 * \brief Extract the first value corresponding to the given configuration element from the parameters.
 * \tparam D The configuration element type
 * \tparam Args The arguments to extract the value from
 */
template <typename D, typename... Args>
struct get_value_1;

/*!
 * \copydoc get_value_1
 */
template <typename D, typename T2, typename... Args>
struct get_value_1<D, T2, Args...> : conditional_constant_v1<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_value_1<D, Args...>> {};

/*!
 * \copydoc get_value_1
 */
template <typename D>
struct get_value_1<D> : std::integral_constant<decltype(D::value_1), D::value_1> {};

/*!
 * \brief Extract the second value corresponding to the given configuration element from the parameters.
 * \tparam D The configuration element type
 * \tparam Args The arguments to extract the value from
 */
template <typename D, typename... Args>
struct get_value_2;

/*!
 * \copydoc get_value_2
 */
template <typename D, typename T2, typename... Args>
struct get_value_2<D, T2, Args...> : conditional_constant_v2<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_value_2<D, Args...>> {};

/*!
 * \copydoc get_value_2
 */
template <typename D>
struct get_value_2<D> : std::integral_constant<decltype(D::value_2), D::value_2> {};

/*!
 * \brief Extract the value corresponding to the given configuration element from the list of type of the parameters.
 * \tparam D The configuration element type
 * \tparam Args The arguments to extract the value from
 */
template <typename D, typename L>
struct get_value_l;

/*!
 * \copydoc get_value_l
 */
template <typename D, typename... T>
struct get_value_l<D, cpp::type_list<T...>> : cpp::auto_constant<get_value<D, T...>> {};

/*!
 * \brief Extract the type corresponding to the given configuration element from
 * the list of the parameters.
 * \tparam D The configuration element type
 * \tparam Args The arguments to extract the type from
 */
template <typename D, typename... Args>
struct get_type;

/*!
 * \copydoc get_type
 */
template <typename D, typename T2, typename... Args>
struct get_type<D, T2, Args...> {
    // Simply using conditional_t is not enough since T2::value can be a real value and not a type and therefore should not always be evaluated

    /*!
     * \brief The extracted value type
     */
    using value = typename cpp::conditional_type_constant_c<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_type<D, Args...>>::type;
};

/*!
 * \copydoc get_type
 */
template <typename D>
struct get_type<D> {
    using value = typename D::value; ///< The extracted value type
};


/*!
 * \brief Extract the type corresponding to the given configuration element from
 * the list of the parameters.
 * \tparam D The configuration element type
 * \tparam Args The arguments to extract the type from
 */
template <typename D, typename... Args>
using get_type_t = typename get_type<D, Args...>::value;

/*!
 * \brief Extract the template type corresponding to the given configuration element from
 * the list of the parameters.
 * \tparam D The configuration element type
 * \tparam Args The arguments to extract the template type from
 */
template <typename D, typename... Args>
struct get_template_type;

/*!
 * \copydoc get_template_type
 */
template <typename D, typename T2, typename... Args>
struct get_template_type<D, T2, Args...> {
    /*!
     * \brief The extracted value template type
     */
    template <typename RBM>
    using value = typename cpp::conditional_template_type_constant_c<std::is_same<typename D::type_id, typename T2::type_id>::value, T2, get_template_type<D, Args...>>::template type<RBM>;
};

/*!
 * \copydoc get_template_type
 */
template <typename D>
struct get_template_type<D> {
    /*!
     * \brief The extracted value template type
     */
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

template <typename Tuple, typename Functor, size_t I1>
void for_each_type_sub(Functor&& functor, const std::index_sequence<I1>& /* i */) {
    functor(static_cast<std::tuple_element_t<I1, Tuple>*>(nullptr));
}

template <typename Tuple, typename Functor, size_t I1, size_t... I, cpp_enable_iff((sizeof...(I) > 0))>
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

template <bool C, typename W, size_t... Dims>
struct conditional_fast_matrix {
    using type = std::conditional_t<
        C,
        etl::fast_matrix<W, Dims...>,
        etl::fast_matrix_impl<W, std::array<W, 0>, etl::order::RowMajor, Dims...>>;
};

template <bool C, typename W, size_t... Dims>
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

/*!
 * \brief Extract the value corresponding to the given configuration element from the list of type of the parameters.
 * \tparam D The configuration element type
 * \tparam Args The arguments to extract the value from
 */
template <typename D, typename L>
constexpr auto get_value_l_v = detail::get_value_l<D, L>::value;

/*!
 * \brief Value traits to compute the addition of all the given values
 */
template <size_t... Dims>
constexpr size_t add_all = (Dims + ...);

} //end of dll namespace
