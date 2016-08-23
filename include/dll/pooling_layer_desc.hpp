//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <std::size_t T_I1, std::size_t T_I2, std::size_t T_I3, std::size_t T_C1, std::size_t T_C2, std::size_t T_C3, typename... Parameters>
struct pooling_layer_3d_desc {
    static constexpr const std::size_t I1 = T_I1;
    static constexpr const std::size_t I2 = T_I2;
    static constexpr const std::size_t I3 = T_I3;
    static constexpr const std::size_t C1 = T_C1;
    static constexpr const std::size_t C2 = T_C2;
    static constexpr const std::size_t C3 = T_C3;

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<double>, Parameters...>::value;

    static_assert(C1 > 0, "Cannot shrink a layer by less than 1");
    static_assert(C2 > 0, "Cannot shrink a layer by less than 1");
    static_assert(C3 > 0, "Cannot shrink a layer by less than 1");
    static_assert(I1 % C1 == 0, "Input dimension is not divisible by C");
    static_assert(I2 % C2 == 0, "Input dimension is not divisible by C");
    static_assert(I3 % C3 == 0, "Input dimension is not divisible by C");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id>, Parameters...>::value,
        "Invalid parameters type for pooling_layer");
};

template <typename... Parameters>
struct dyn_pooling_layer_3d_desc {
    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<double>, Parameters...>::value;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id>, Parameters...>::value,
        "Invalid parameters type for pooling_layer");
};

} //end of dll namespace
