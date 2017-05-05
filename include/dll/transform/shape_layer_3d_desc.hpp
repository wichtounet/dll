//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <size_t C_T, size_t H_T, size_t W_T, typename... Parameters>
struct shape_layer_3d_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    static constexpr size_t C = C_T; ///< The size
    static constexpr size_t H = H_T; ///< The size
    static constexpr size_t W = W_T; ///< The size

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*!
     * The layer type
     */
    using layer_t = shape_layer_3d<shape_layer_3d_desc<C_T, H_T, W_T, Parameters...>>;

    /*!
     * The dynamic layer type
     */
    using dyn_layer_t = dyn_shape_layer_3d<dyn_shape_layer_3d_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id>, Parameters...>::value,
        "Invalid parameters type for shape_layer_3d_desc");
};

} //end of dll namespace
