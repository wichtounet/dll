//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_PARALLEL_HPP
#define DLL_PARALLEL_HPP

#include "cpp_utils/parallel.hpp"             //Parallel

namespace dll {

template<bool Parallel>
struct thread_pool {
    //Does not do anythin by default
};

template<>
struct thread_pool<true> : cpp::default_thread_pool<> {
    //Simply inherits from default thread pool
};

template<typename Container, typename Functor>
void maybe_parallel_foreach_i(thread_pool<true>& thread_pool, const Container& container, Functor&& fun){
    parallel_foreach_i(thread_pool, container, std::forward<Functor>(fun));
}

template<typename Iterator, typename Functor>
void maybe_parallel_foreach_i(thread_pool<true>& thread_pool, Iterator it, Iterator end, Functor&& fun){
    parallel_foreach_i(thread_pool, it, end, std::forward<Functor>(fun));
}

template<typename Iterator, typename Iterator2, typename Functor>
void maybe_parallel_foreach_pair_i(thread_pool<true>& thread_pool, Iterator it, Iterator end, Iterator2 iit, Iterator2 ilast, Functor&& fun){
    parallel_foreach_pair_i(thread_pool, it, end, iit, ilast, std::forward<Functor>(fun));
}

template<typename Container, typename Functor>
void maybe_parallel_foreach_i(thread_pool<false>& /*thread_pool*/, const Container& container, Functor&& fun){
    for(std::size_t i = 0; i < container.size(); ++i){
        fun(container[i], i);
    }
}

template<typename Iterator, typename Functor>
void maybe_parallel_foreach_i(thread_pool<false>& /*thread_pool*/, Iterator it, Iterator end, Functor&& fun){
    for(std::size_t i = 0; it != end; ++it, ++i){
        fun(*it, i);
    }
}

template<typename Iterator, typename Iterator2, typename Functor>
void maybe_parallel_foreach_pair_i(thread_pool<false>& /*thread_pool*/, Iterator it, Iterator end, Iterator2 iit, Iterator2 ilast, Functor&& fun){
    cpp_unused(ilast);

    for(std::size_t i = 0; it != end; ++it, ++iit, ++i){
        fun(*it, *iit, i);
    }
}

} //end of dll namespace

#endif
