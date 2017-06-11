//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Descriptor for a Dynamic 1D shaping layer.
 */
template<typename... Parameters>
struct dyn_shape_layer_1d_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*! The layer type */
    using layer_t = dyn_shape_layer_1d<dyn_shape_layer_1d_desc<Parameters...>>;

    /*! The layer type */
    using dyn_layer_t = dyn_shape_layer_1d<dyn_shape_layer_1d_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id>, Parameters...>::value,
        "Invalid parameters type for dyn_shape_layer_1d_desc");
};

} //end of dll namespace
