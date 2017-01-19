//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <typename... Parameters>
struct activation_layer_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    static constexpr const function activation_function = detail::get_value<activation<function::SIGMOID>, Parameters...>::value;

    /*!
     * The layer type
     */
    using layer_t = activation_layer<activation_layer_desc<Parameters...>>;

    /*!
     * The dynamic layer type
     */
    using dyn_layer_t = activation_layer<activation_layer_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<activation_id>, Parameters...>::value,
        "Invalid parameters type for activation_layer_desc");
};

} //end of dll namespace
