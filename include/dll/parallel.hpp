//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_PARALLEL_HPP
#define DLL_PARALLEL_HPP

namespace dll {

#ifdef DLL_PARALLEL

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

#endif //DLL_PARALLEL

} //end of dll namespace

#endif