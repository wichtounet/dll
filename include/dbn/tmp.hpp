//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_TMP_HPP
#define DBN_TMP_HPP

template< bool B, class T = void >
using disable_if_t = typename std::enable_if<!B, T>::type;

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

namespace dbn {

template<typename T1, typename... Args>
struct is_present;

template<typename T1, typename T2, typename... Args>
struct is_present<T1, T2, Args...> {
    static constexpr const bool value = std::is_same<T1, T2>::value || is_present<T1, Args...>::value;
};

template<typename T1>
struct is_present<T1> {
    static constexpr const bool value = false;
};

template<typename... Valid>
struct tmp_list {
    template<typename T, typename Enable = void>
    struct check {
        static constexpr const bool value = is_present<T, Valid...>::value;
    };

    template<typename T>
    struct check<T, enable_if_t<T::marker>> {
        static constexpr const bool value = is_present<typename T::type, Valid...>::value;;
    };
};

template<typename V, typename... Args>
struct is_valid;

template<typename V, typename T1, typename... Args>
struct is_valid <V, T1, Args...> {
    static constexpr const bool value = V::template check<T1>::value && is_valid<V, Args...>::value;
};

template<typename V>
struct is_valid <V> {
    static constexpr const bool value = true;
};

template<typename D, typename... Args>
struct get_value {};

template<typename D, typename T2, typename... Args>
struct get_value<D, T2, Args...> {
    template<typename D2, typename T22, typename Enable = void>
    struct get_value_2 {
        static constexpr const auto value = get_value<D, Args...>::value;
    };

    template<typename D2, typename T22>
    struct get_value_2 <D2, T22, enable_if_t<std::is_same<typename D2::type, typename T22::type>::value>> {
        static constexpr const auto value = T22::value;
    };

    static constexpr const auto value = get_value_2<D, T2>::value;
};

template<typename D>
struct get_value<D> {
    static constexpr const auto value = D::value;
};

} //end of dbn namespace

#endif