//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_UTILS_HPP
#define DBN_UTILS_HPP

template< bool B, class T = void >
using disable_if_t = typename std::enable_if<!B, T>::type;

namespace dbn {

template<size_t I, class Tuple, typename F>
struct for_each_impl {
    static void for_each(Tuple& t, F&& f) {
        for_each_impl<I - 1, Tuple, F>::for_each(t, std::forward<F>(f));
        f(std::get<I>(t));
    }

    static void for_each_i(Tuple& t, F&& f) {
        for_each_impl<I - 1, Tuple, F>::for_each_i(t, std::forward<F>(f));
        f(I, std::get<I>(t));
    }
};

template<class Tuple, typename F>
struct for_each_impl<0, Tuple, F> {
    static void for_each(Tuple& t, F&& f) {
        f(std::get<0>(t));
    }

    static void for_each_i(Tuple& t, F&& f) {
        f(0, std::get<0>(t));
    }
};

template<class Tuple, typename F>
void for_each(Tuple& t, F&& f) {
    for_each_impl<std::tuple_size<Tuple>::value - 1, Tuple, F>::for_each(t, std::forward<F>(f));
}

template<class Tuple, typename F>
void for_each_i(Tuple& t, F&& f) {
    for_each_impl<std::tuple_size<Tuple>::value - 1, Tuple, F>::for_each_i(t, std::forward<F>(f));
}

} //end of namespace dbn

#endif
