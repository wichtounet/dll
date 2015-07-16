//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_CONTEXT_HPP
#define DLL_CONTEXT_HPP

namespace dll {

template<template<typename> class Context, typename T>
struct context_builder;

template<template<typename> class Context, typename... Args>
struct context_builder<Context, std::tuple<Args...>> {
    using type = std::tuple<Context<Args>...>;
};

template<template<typename,typename> class Context, typename DBN, typename T>
struct dbn_context_builder;

template<template<typename,typename> class Context, typename DBN, typename... Args>
struct dbn_context_builder<Context, DBN, std::tuple<Args...>> {
    using type = std::tuple<Context<DBN, Args>...>;
};

} //end of dll namespace

#endif
