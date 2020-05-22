//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_conf.hpp"
#include "dll/util/tmp.hpp"

namespace dll {

/*!
 * \brief Describe a standard embedding layer.
 */
template <size_t V_T, size_t I_T, size_t K_T, typename... Parameters>
struct embedding_layer_desc {
    static constexpr size_t V   = V_T;  ///< The size of the vocabulary
    static constexpr size_t I   = I_T;  ///< The size of each input
    static constexpr size_t K   = K_T;  ///< The size of embeddings

    /*!
     * \brief A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    using w_initializer = detail::get_type_t<initializer<init_uniform<constant(-1.0), constant(1.0)>>, Parameters...>;     ///< The initializer for the weights

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*! The embedding type */
    using layer_t = embedding_layer_impl<embedding_layer_desc<V, I, K, Parameters...>>;

    /*! The embedding type */
    using dyn_layer_t = dyn_embedding_layer_impl<dyn_embedding_layer_desc<Parameters...>>;

    static_assert(V > 0, "At least one char in vocabulary is necessary");
    static_assert(I > 0, "At least one input is necessary");
    static_assert(K > 0, "At least one embedding is necessary");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<weight_type_id, initializer_id>, Parameters...>,
        "Invalid parameters type for embedding_layer_desc");
};

/*!
 * \brief Describe a standard embedding layer.
 */
template <size_t V_T, size_t I_T, size_t K_T, typename... Parameters>
using embedding_layer = typename embedding_layer_desc<V_T, I_T, K_T, Parameters...>::layer_t;

} //end of dll namespace
