//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DENSE_DESC_HPP
#define DLL_DENSE_DESC_HPP

#include "base_conf.hpp"
#include "tmp.hpp"

namespace dll {

/*!
 * \brief Describe a dense layer.
 */
template <std::size_t visibles, std::size_t hiddens, typename... Parameters>
struct dense_desc {
    static constexpr const std::size_t num_visible = visibles;
    static constexpr const std::size_t num_hidden  = hiddens;

    using parameters = cpp::type_list<Parameters...>;

    static constexpr const function activation_function = detail::get_value<activation<function::SIGMOID>, Parameters...>::value;

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*! The dense type */
    using layer_t = dense_layer<dense_desc<visibles, hiddens, Parameters...>>;

    static_assert(num_visible > 0, "There must be at least 1 visible unit");
    static_assert(num_hidden > 0, "There must be at least 1 hidden unit");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id, dbn_only_id, activation_id>, Parameters...>::value,
        "Invalid parameters type for dense_desc");
};

} //end of dll namespace

#endif
