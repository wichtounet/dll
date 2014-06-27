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

namespace dbn {

struct conf_elt {
    static constexpr const bool marker = true;
};

struct batch_size_id;
struct visible_unit_id;
struct hidden_unit_id;
struct weight_decay_id;

template<std::size_t B>
struct batch_size : conf_elt {
    using type = batch_size_id;

    static constexpr const bool marker = true;
    static constexpr const std::size_t value = B;
};

template<Type VT>
struct visible_unit : conf_elt {
    using type = visible_unit_id;

    static constexpr const Type value = VT;
};

template<Type HT>
struct hidden_unit : conf_elt  {
    using type = hidden_unit_id;

    static constexpr const Type value = HT;
};

template<DecayType T>
struct weight_decay : conf_elt  {
    using type = weight_decay_id;

    static constexpr const DecayType value = T;
};

template<template<typename> class T>
struct trainer : conf_elt  {
    template <typename RBM>
    using value = T<RBM>;
};

struct momentum {};
struct sparsity {};
struct debug {};
struct init_weights {};
struct in_dbn {};

} //end of dbn namespace

#endif