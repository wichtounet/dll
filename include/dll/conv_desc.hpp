//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_CONV_DESC_HPP
#define DLL_CONV_DESC_HPP

#include "base_conf.hpp"
#include "util/tmp.hpp"

namespace dll {

/*!
 * \brief Describe a convolutional layer.
 */
template <std::size_t NC_T, std::size_t NV_1, std::size_t NV_2, std::size_t K_T, std::size_t NH_1, std::size_t NH_2, typename... Parameters>
struct conv_desc {
    static constexpr const std::size_t NV1 = NV_1;
    static constexpr const std::size_t NV2 = NV_2;
    static constexpr const std::size_t NH1 = NH_1;
    static constexpr const std::size_t NH2 = NH_2;
    static constexpr const std::size_t NC  = NC_T;
    static constexpr const std::size_t K   = K_T;

    using parameters = cpp::type_list<Parameters...>;

    static constexpr const function activation_function = detail::get_value<activation<function::SIGMOID>, Parameters...>::value;

    /*! The type used to store the weights */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    /*! The conv type */
    using layer_t = conv_layer<conv_desc<NC_T, NV_1, NV_2, K_T, NH_1, NH_2, Parameters...>>;

    static_assert(NV1 > 0, "A matrix of at least 1x1 is necessary for the visible units");
    static_assert(NV2 > 0, "A matrix of at least 1x1 is necessary for the visible units");
    static_assert(NH1 > 0, "A matrix of at least 1x1 is necessary for the hidden units");
    static_assert(NH2 > 0, "A matrix of at least 1x1 is necessary for the hidden units");
    static_assert(NC > 0, "At least one channel is necessary");
    static_assert(K > 0, "At least one group is necessary");

    static_assert(NV1 >= NH1, "The convolutional filter must be of at least size 1");
    static_assert(NV2 >= NH2, "The convolutional filter must be of at least size 1");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id, dbn_only_id, activation_id>, Parameters...>::value,
        "Invalid parameters type for rbm_desc");
};

} //end of dll namespace

#endif
