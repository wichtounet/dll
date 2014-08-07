//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_TUPLE_UTILS_HPP
#define DLL_TUPLE_UTILS_HPP

namespace dll {

namespace detail {

template<size_t I, typename Tuple, typename F>
struct for_each_impl {
    static void for_each(Tuple& t, F&& f) {
        for_each_impl<I - 1, Tuple, F>::for_each(t, std::forward<F>(f));
        f(std::get<I>(t));
    }

    static void for_each_i(Tuple& t, F&& f) {
        for_each_impl<I - 1, Tuple, F>::for_each_i(t, std::forward<F>(f));
        f(I, std::get<I>(t));
    }

    static void for_each_pair(Tuple& t, F&& f) {
        for_each_impl<I - 1, Tuple, F>::for_each_pair(t, std::forward<F>(f));
        f(std::get<I>(t), std::get<I+1>(t));
    }

    static void for_each_pair_i(Tuple& t, F&& f) {
        for_each_impl<I - 1, Tuple, F>::for_each_pair_i(t, std::forward<F>(f));
        f(I, std::get<I>(t), std::get<I+1>(t));
    }

    static void for_each_rpair(Tuple& t, F&& f) {
        f(std::get<I>(t), std::get<I+1>(t));
        for_each_impl<I - 1, Tuple, F>::for_each_rpair(t, std::forward<F>(f));
    }

    static void for_each_rpair_i(Tuple& t, F&& f) {
        f(I, std::get<I>(t), std::get<I+1>(t));
        for_each_impl<I - 1, Tuple, F>::for_each_rpair_i(t, std::forward<F>(f));
    }
};

template<typename Tuple, typename F>
struct for_each_impl<0, Tuple, F> {
    static void for_each(Tuple& t, F&& f) {
        f(std::get<0>(t));
    }

    static void for_each_i(Tuple& t, F&& f) {
        f(0, std::get<0>(t));
    }

    static void for_each_pair(Tuple& t, F&& f) {
        f(std::get<0>(t), std::get<1>(t));
    }

    static void for_each_pair_i(Tuple& t, F&& f) {
        f(0, std::get<0>(t), std::get<1>(t));
    }

    static void for_each_rpair(Tuple& t, F&& f) {
        f(std::get<0>(t), std::get<1>(t));
    }

    static void for_each_rpair_i(Tuple& t, F&& f) {
        f(0, std::get<0>(t), std::get<1>(t));
    }
};

template<size_t I, typename Tuple1, typename Tuple2, typename F>
struct dual_for_each_impl {
    static void for_each(Tuple1& t1, Tuple2& t2, F&& f) {
        dual_for_each_impl<I - 1, Tuple1, Tuple2, F>::for_each(t1, t2, std::forward<F>(f));
        f(std::get<I>(t1), std::get<I>(t2));
    }

    static void for_each_pair(Tuple1& t1, Tuple2& t2, F&& f) {
        dual_for_each_impl<I - 1, Tuple1, Tuple2, F>::for_each_pair(t1, t2, std::forward<F>(f));
        f(std::get<I>(t1), std::get<I+1>(t1), std::get<I>(t2), std::get<I+1>(t2));
    }

    static void for_each_rpair_i(Tuple1& t1, Tuple2& t2, F&& f) {
        f(I, std::get<I>(t1), std::get<I+1>(t1), std::get<I>(t2), std::get<I+1>(t2));
        dual_for_each_impl<I - 1, Tuple1, Tuple2, F>::for_each_rpair_i(t1, t2, std::forward<F>(f));
    }
};

template<typename Tuple1, typename Tuple2, typename F>
struct dual_for_each_impl<0, Tuple1, Tuple2, F> {
    static void for_each(Tuple1& t1, Tuple2& t2, F&& f) {
        f(std::get<0>(t1), std::get<0>(t2));
    }

    static void for_each_pair(Tuple1& t1, Tuple2& t2, F&& f) {
        f(std::get<0>(t1), std::get<1>(t1), std::get<0>(t2), std::get<1>(t2));
    }

    static void for_each_rpair_i(Tuple1& t1, Tuple2& t2, F&& f) {
        f(0, std::get<0>(t1), std::get<1>(t1), std::get<0>(t2), std::get<1>(t2));
    }
};

//Normal versions

template<typename Tuple, typename F>
void for_each(Tuple& t, F&& f) {
    for_each_impl<std::tuple_size<Tuple>::value - 1, Tuple, F>::for_each(t, std::forward<F>(f));
}

template<typename Tuple, typename F>
void for_each_i(Tuple& t, F&& f) {
    for_each_impl<std::tuple_size<Tuple>::value - 1, Tuple, F>::for_each_i(t, std::forward<F>(f));
}

template<typename Tuple, typename F>
void for_each_pair(Tuple& t, F&& f) {
    if(std::tuple_size<Tuple>::value > 1){
        for_each_impl<std::tuple_size<Tuple>::value - 2, Tuple, F>::for_each_pair(t, std::forward<F>(f));
    }
}

template<typename Tuple, typename F>
void for_each_pair_i(Tuple& t, F&& f) {
    if(std::tuple_size<Tuple>::value > 1){
        for_each_impl<std::tuple_size<Tuple>::value - 2, Tuple, F>::for_each_pair_i(t, std::forward<F>(f));
    }
}

template<typename Tuple, typename F>
void for_each_rpair(Tuple& t, F&& f) {
    if(std::tuple_size<Tuple>::value > 1){
        for_each_impl<std::tuple_size<Tuple>::value - 2, Tuple, F>::for_each_rpair(t, std::forward<F>(f));
    }
}

template<typename Tuple, typename F>
void for_each_rpair_i(Tuple& t, F&& f) {
    if(std::tuple_size<Tuple>::value > 1){
        for_each_impl<std::tuple_size<Tuple>::value - 2, Tuple, F>::for_each_rpair_i(t, std::forward<F>(f));
    }
}

//Dual versions

template<typename Tuple1, typename Tuple2, typename F>
void for_each(Tuple1& t1, Tuple2& t2, F&& f) {
    static_assert(std::tuple_size<Tuple1>::value == std::tuple_size<Tuple2>::value, "Can only iterate tuples of same size");

    dual_for_each_impl<std::tuple_size<Tuple1>::value - 1, Tuple1, Tuple2, F>::for_each(t1, t2, std::forward<F>(f));
}

template<typename Tuple1, typename Tuple2, typename F>
void for_each_pair(Tuple1& t1, Tuple2& t2, F&& f) {
    static_assert(std::tuple_size<Tuple1>::value == std::tuple_size<Tuple2>::value, "Can only iterate tuples of same size");

    if(std::tuple_size<Tuple1>::value > 1){
        dual_for_each_impl<std::tuple_size<Tuple1>::value - 2, Tuple1, Tuple2, F>::for_each_pair(t1, t2, std::forward<F>(f));
    }
}

template<typename Tuple1, typename Tuple2, typename F>
void for_each_rpair_i(Tuple1& t1, Tuple2& t2, F&& f) {
    static_assert(std::tuple_size<Tuple1>::value == std::tuple_size<Tuple2>::value, "Can only iterate tuples of same size");

    if(std::tuple_size<Tuple1>::value > 1){
        dual_for_each_impl<std::tuple_size<Tuple1>::value - 2, Tuple1, Tuple2, F>::for_each_rpair_i(t1, t2, std::forward<F>(f));
    }
}

} //end of namespace detail

} //end of namespace dll

#endif
