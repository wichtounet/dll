//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_conf.hpp"
#include "dll/util/tmp.hpp"

namespace dll {

/*!
 * \brief Describe a dense layer.
 */
template <std::size_t visibles, std::size_t hiddens, typename... Parameters>
struct dense_desc {
    static constexpr const std::size_t num_visible = visibles;
    static constexpr const std::size_t num_hidden  = hiddens;

    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    static constexpr auto activation_function = detail::get_value<activation<function::SIGMOID>, Parameters...>::value;
    static constexpr auto initializer_t       = detail::get_value<initializer<initializer_type::LECUN>, Parameters...>::value;

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*! The dense type */
    using layer_t = dense_layer<dense_desc<visibles, hiddens, Parameters...>>;

    /*! The dense type */
    using dyn_layer_t = dyn_dense_layer<dyn_dense_desc<Parameters...>>;

    static_assert(num_visible > 0, "There must be at least 1 visible unit");
    static_assert(num_hidden > 0, "There must be at least 1 hidden unit");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id, dbn_only_id, activation_id, initializer_id>, Parameters...>::value,
        "Invalid parameters type for dense_desc");
};

} //end of dll namespace
