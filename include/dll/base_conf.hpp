//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_BASE_CONF_HPP
#define DLL_BASE_CONF_HPP

#include <cstddef>

#include "unit_type.hpp"
#include "decay_type.hpp"
#include "sparsity_method.hpp"
#include "bias_mode.hpp"

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
struct sparsity_id;
struct bias_id;

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

/*!
 * \brief Activate sparsity and select the method to use
 */
template<sparsity_method M = sparsity_method::GLOBAL_TARGET>
struct sparsity : conf_elt {
    using type = sparsity_id;

    static constexpr const sparsity_method value = M;
};

/*!
 * \brief Select the bias method
 */
template<bias_mode M = bias_mode::SIMPLE>
struct bias : conf_elt, std::integral_constant<bias_mode, M> {
    using type = bias_id;
};

template<template<typename> class T>
struct trainer : conf_elt  {
    using type = trainer_id;

    template <typename RBM>
    using value = T<RBM>;
};

template<template<typename...> class T>
struct watcher : conf_elt  {
    using type = watcher_id;

    template <typename RBM>
    using value = T<RBM>;
};

struct momentum {};
struct init_weights {};

} //end of dbn namespace

#endif