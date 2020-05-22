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
 * \brief Descriptor for a dense layer
 */
template <size_t visibles, size_t hiddens, typename... Parameters>
struct dense_layer_desc {
    static constexpr size_t num_visible = visibles; ///< The number of visible units of the dense layer
    static constexpr size_t num_hidden  = hiddens;  ///< The number of hidden units of the dense layer

    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    static constexpr auto activation_function = detail::get_value_v<activation<function::SIGMOID>, Parameters...>;            ///< The layer's activation function

    using w_initializer = detail::get_type_t<initializer<init_lecun>, Parameters...>;     ///< The initializer for the weights
    using b_initializer = detail::get_type_t<initializer_bias<init_zero>, Parameters...>; ///< The initializer for the biases

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*! The dense type */
    using layer_t = dense_layer_impl<dense_layer_desc<visibles, hiddens, Parameters...>>;

    /*! The dense type */
    using dyn_layer_t = dyn_dense_layer_impl<dyn_dense_layer_desc<Parameters...>>;

    static_assert(num_visible > 0, "There must be at least 1 visible unit");
    static_assert(num_hidden > 0, "There must be at least 1 hidden unit");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<
            weight_type_id, activation_id, initializer_id, initializer_bias_id, no_bias_id>,
            Parameters...>,
        "Invalid parameters type for dense_layer_desc");
};

/*!
 * \brief Describe a dense layer
 */
template <size_t visibles, size_t hiddens, typename... Parameters>
using dense_layer = typename dense_layer_desc<visibles, hiddens, Parameters...>::layer_t;

} //end of dll namespace
