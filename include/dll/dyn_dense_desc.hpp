//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "base_conf.hpp"
#include "util/tmp.hpp"

namespace dll {

/*!
 * \brief Describe a dense layer.
 */
template <typename... Parameters>
struct dyn_dense_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    static constexpr const function activation_function = detail::get_value<activation<function::SIGMOID>, Parameters...>::value;

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*! The dense type */
    using layer_t = dyn_dense_layer<dyn_dense_desc<Parameters...>>;

    /*! The dense type */
    using dyn_layer_t = dyn_dense_layer<dyn_dense_desc<Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id, dbn_only_id, activation_id>, Parameters...>::value,
        "Invalid parameters type for dense_desc");
};

} //end of dll namespace
