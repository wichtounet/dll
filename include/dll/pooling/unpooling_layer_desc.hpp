//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Descriptor for an unpooling 3D layer
 */
template <size_t T_I1, size_t T_I2, size_t T_I3, size_t T_C1, size_t T_C2, size_t T_C3, typename... Parameters>
struct unpooling_3d_layer_desc {
    static constexpr size_t I1 = T_I1; ///< The input first dimension
    static constexpr size_t I2 = T_I2; ///< The input second dimension
    static constexpr size_t I3 = T_I3; ///< The input third dimension
    static constexpr size_t C1 = T_C1; ///< The pooling first dimension
    static constexpr size_t C2 = T_C2; ///< The pooling second dimension
    static constexpr size_t C3 = T_C3; ///< The pooling third dimension

    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    static_assert(C1 > 0, "Cannot shrink a layer by less than 1");
    static_assert(C2 > 0, "Cannot shrink a layer by less than 1");
    static_assert(C3 > 0, "Cannot shrink a layer by less than 1");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<weight_type_id>, Parameters...>,
        "Invalid parameters type for unpooling_layer");
};

/*!
 * \brief Descriptor for a dynamic unpooling 3D layer
 */
template <typename... Parameters>
struct dyn_unpooling_3d_layer_desc {
    /*! The type used to store the weights */
    using weight = detail::get_type_t<weight_type<float>, Parameters...>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<cpp::type_list<weight_type_id>, Parameters...>,
        "Invalid parameters type for dyn_unpooling_layer");
};

} //end of dll namespace
