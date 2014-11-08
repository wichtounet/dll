//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_PARALLEL_HPP
#define DLL_PARALLEL_HPP

#include "cpp_utils/parallel.hpp"             //Parallel

namespace dll {

#ifdef DLL_PARALLEL

using thread_pool = cpp::default_thread_pool<>;

template<typename TP, typename Container, typename Functor>
void maybe_parallel_foreach_i(TP& thread_pool, const Container& container, Functor&& fun){
    parallel_foreach_i(thread_pool, container, std::forward<Functor>(fun));
}

template<typename TP, typename Iterator, typename Functor>
void maybe_parallel_foreach_i(TP& thread_pool, Iterator it, Iterator end, Functor&& fun){
    parallel_foreach_i(thread_pool, it, end, std::forward<Functor>(fun));
}

template<typename TP, typename Iterator, typename Iterator2, typename Functor>
void maybe_parallel_foreach_pair_i(TP& thread_pool, Iterator it, Iterator end, Iterator2 iit, Iterator2 ilast, Functor&& fun){
    parallel_foreach_pair_i(thread_pool, it, end, iit, ilast, std::forward<Functor>(fun));
}

#else //!DLL_PARALLEL

struct fake_thread_pool {};

using thread_pool = fake_thread_pool;

template<typename TP, typename Container, typename Functor>
void maybe_parallel_foreach_i(TP& /*thread_pool*/, const Container& container, Functor&& fun){
    for(std::size_t i = 0; i < container.size(); ++i){
        fun(container[i], i);
    }
}

template<typename TP, typename Iterator, typename Functor>
void maybe_parallel_foreach_i(TP& /*thread_pool*/, Iterator it, Iterator end, Functor&& fun){
    for(std::size_t i = 0; it != end; ++it, ++i){
        fun(*it, i);
    }
}

template<typename TP, typename Iterator, typename Iterator2, typename Functor>
void maybe_parallel_foreach_pair_i(TP& /*thread_pool*/, Iterator it, Iterator end, Iterator2 iit, Iterator2 ilast, Functor&& fun){
    cpp_unused(ilast);

    for(std::size_t i = 0; it != end; ++it, ++iit, ++i){
        fun(*it, *iit, i);
    }
}

#endif //DLL_PARALLEL

} //end of dll namespace

#endif
