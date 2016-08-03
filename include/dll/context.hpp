//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <template <typename> class Context, typename T>
struct context_builder;

template <template <typename> class Context, typename... Args>
struct context_builder<Context, std::tuple<Args...>> {
    using type = std::tuple<Context<Args>...>;
};

template <template <typename...> class Context, typename DBN, typename T>
struct dbn_context_builder;

template <template <typename...> class Context, typename DBN, typename... Args>
struct dbn_context_builder<Context, DBN, std::tuple<Args...>> {
    using type = std::tuple<Context<DBN, Args>...>;
};

template <template <typename...> class Context, typename DBN, typename T>
struct dbn_context_builder_i_impl;

template <template <typename...> class Context, typename DBN, std::size_t... I>
struct dbn_context_builder_i_impl<Context, DBN, std::index_sequence<I...>> {
    using type = std::tuple<Context<DBN, typename DBN::template layer_type<I>>...>;
};

template <template <typename...> class Context, typename DBN>
struct dbn_context_builder_i {
    using type = typename dbn_context_builder_i_impl<Context, DBN, std::make_index_sequence<DBN::layers>>::type;
};

} //end of dll namespace
