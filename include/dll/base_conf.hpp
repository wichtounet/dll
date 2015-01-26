//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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

template<typename ID>
struct basic_conf_elt {
    using type_id = ID;
};

template<typename ID, typename T>
struct type_conf_elt {
    using type_id = ID;

    using value = T;
};

template<typename ID, template<typename...> class T>
struct template_type_conf_elt {
    using type_id = ID;

    template <typename RBM>
    using value = T<RBM>;
};

template<typename ID, typename T, T value>
struct value_conf_elt : std::integral_constant<T, value> {
    using type_id = ID;
};

struct batch_size_id;
struct visible_id;
struct hidden_id;
struct pooling_id;
struct weight_decay_id;
struct trainer_id;
struct watcher_id;
struct sparsity_id;
struct bias_id;
struct momentum_id;
struct parallel_id;
struct shuffle_id;
struct svm_concatenate_id;
struct svm_scale_id;
struct init_weights_id;
struct weight_type_id;
struct free_energy_id;

template<std::size_t B>
struct batch_size : value_conf_elt<batch_size_id, std::size_t, B> {};

template<unit_type VT>
struct visible : value_conf_elt<visible_id, unit_type, VT> {};

template<unit_type HT>
struct hidden : value_conf_elt<hidden_id, unit_type, HT> {};

template<unit_type PT>
struct pooling : value_conf_elt<pooling_id, unit_type, PT> {};

template<decay_type T = decay_type::L2>
struct weight_decay : value_conf_elt<weight_decay_id, decay_type, T> {};

/*!
 * \brief Activate sparsity and select the method to use
 */
template<sparsity_method M = sparsity_method::GLOBAL_TARGET>
struct sparsity : value_conf_elt<sparsity_id, sparsity_method, M> {};

/*!
 * \brief Select the bias method
 */
template<bias_mode M = bias_mode::SIMPLE>
struct bias : value_conf_elt<bias_id, bias_mode, M>{};

template<typename T>
struct weight_type : type_conf_elt<weight_type_id, T> {};

template<template<typename...> class T>
struct trainer : template_type_conf_elt<trainer_id, T> {};

template<template<typename...> class T>
struct watcher : template_type_conf_elt<watcher_id, T> {};

struct momentum : basic_conf_elt<momentum_id> {};
struct parallel : basic_conf_elt<parallel_id> {};
struct svm_concatenate : basic_conf_elt<svm_concatenate_id> {};
struct svm_scale : basic_conf_elt<svm_scale_id> {};
struct init_weights : basic_conf_elt<init_weights_id> {};
struct shuffle : basic_conf_elt<shuffle_id> {};
struct free_energy : basic_conf_elt<free_energy_id> {};

} //end of dll namespace

#endif
