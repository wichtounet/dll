//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Descriptor for a 3D shaping layer.
 */
template <size_t C_T, size_t H_T, size_t W_T, typename... Parameters>
struct shape_3d_layer_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    static constexpr size_t C = C_T; ///< The size
    static constexpr size_t H = H_T; ///< The size
    static constexpr size_t W = W_T; ///< The size

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    /*!
     * The layer type
     */
    using layer_t = shape_3d_layer_impl<shape_3d_layer_desc<C_T, H_T, W_T, Parameters...>>;

    /*!
     * The dynamic layer type
     */
    using dyn_layer_t = dyn_shape_3d_layer_impl<dyn_shape_3d_layer_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<weight_type_id>, Parameters...>,
        "Invalid parameters type for shape_3d_layer_desc");
};

/*!
 * \brief Descriptor for a 3D shaping layer.
 */
template <size_t C_T, size_t H_T, size_t W_T, typename... Parameters>
using shape_3d_layer = typename shape_3d_layer_desc<C_T, H_T, W_T, Parameters...>::layer_t;

} //end of dll namespace
