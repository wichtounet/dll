//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_BASE_CONF_HPP
#define DBN_BASE_CONF_HPP

#include <cstddef>

#include "unit_type.hpp"
#include "decay_type.hpp"

namespace dll {

struct conf_elt {
    static constexpr const bool marker = true;
};

struct batch_size_id;
struct visible_id;
struct hidden_id;
struct pooling_unit_id;
struct weight_decay_id;
struct trainer_id;
struct watcher_id;

template<std::size_t B>
struct batch_size : conf_elt {
    using type = batch_size_id;

    static constexpr const std::size_t value = B;
};

template<unit_type VT>
struct visible : conf_elt {
    using type = visible_id;

    static constexpr const unit_type value = VT;
};

template<unit_type HT>
struct hidden : conf_elt  {
    using type = hidden_id;

    static constexpr const unit_type value = HT;
};

template<unit_type PT>
struct pooling_unit : conf_elt  {
    using type = pooling_unit_id;

    static constexpr const unit_type value = PT;
};

template<decay_type T>
struct weight_decay : conf_elt  {
    using type = weight_decay_id;

    static constexpr const decay_type value = T;
};

template<template<typename> class T>
struct trainer : conf_elt  {
    using type = trainer_id;

    template <typename RBM>
    using value = T<RBM>;
};

template<template<typename> class T>
struct watcher : conf_elt  {
    using type = watcher_id;

    template <typename RBM>
    using value = T<RBM>;
};

struct momentum {};
struct sparsity {};
struct init_weights {};
struct in_dbn {};

} //end of dbn namespace

#endif