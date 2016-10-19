//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <typename... Parameters>
struct dyn_patches_layer_padh_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*! The layer type */
    using layer_t = dyn_patches_layer_padh<dyn_patches_layer_padh_desc<Parameters...>>;

    /*! The layer type */
    using dyn_layer_t = dyn_patches_layer_padh<dyn_patches_layer_padh_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id>, Parameters...>::value,
        "Invalid parameters type for dyn_patches_layer_pad");
};

} //end of dll namespace
