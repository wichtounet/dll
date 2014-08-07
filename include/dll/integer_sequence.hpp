//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*! \file C++14 integer_sequence */

#ifndef DLL_INTEGER_SEQUENCE_HPP
#define DLL_INTEGER_SEQUENCE_HPP

//TODO Once libstdc++ has been updated to c++14, this should go away
//or if we use only libc++

#include <cstddef>

namespace dll {

template<typename T, T... I>
struct integer_sequence {
  using value_type = T;

  static constexpr std::size_t size() noexcept {
    return sizeof...(I);
  }
};

namespace detail {

template<typename, typename> 
struct concat;

template<typename T, T... A, T... B>
struct concat<integer_sequence<T, A...>, integer_sequence<T, B...>> {
    typedef integer_sequence<T, A..., B...> type;
};

template <typename T, int First, int Count>
struct build_helper {
    using type = typename concat<
            typename build_helper<T, First,           Count/2>::type,
            typename build_helper<T, First + Count/2, Count - Count/2>::type
        >::type;
};

template <typename T, int First>
struct build_helper<T, First, 1> {
    using type = integer_sequence<T, T(First)>;
};

template <typename T, int First>
struct build_helper<T, First, 0> {
    using type = integer_sequence<T>;
};

template <typename T, T N>
using builder = typename build_helper<T, 0, N>::type;

} //end of detail

template <typename T, T N>
using make_integer_sequence = detail::builder<T, N>;

template <std::size_t... I>
using index_sequence = integer_sequence<std::size_t, I...>;

template<size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

} //end of dll namespace

#endif