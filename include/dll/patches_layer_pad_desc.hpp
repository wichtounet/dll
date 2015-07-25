//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_PATCHES_LAYER_DESC_PAD_HPP
#define DLL_PATCHES_LAYER_DESC_PAD_HPP

namespace dll {

template<std::size_t W_T, std::size_t H_T, std::size_t VS_T, std::size_t HS_T, std::size_t Filler_T, typename... Parameters>
struct patches_layer_padh_desc {
    static constexpr const std::size_t width = W_T;
    static constexpr const std::size_t height = H_T;
    static constexpr const std::size_t v_stride = VS_T;
    static constexpr const std::size_t h_stride = HS_T;
    static constexpr const std::size_t filler = Filler_T;

    using weight = typename detail::get_type<weight_type<double>, Parameters...>::value;

    static_assert(width > 0, "A patch must be at least 1 pixel wide");
    static_assert(height > 0, "A patch must be at least 1 pixel high");
    static_assert(v_stride > 0, "The stride is at least 1");
    static_assert(h_stride > 0, "The stride is at least 1");

    /*! The layer type */
    using layer_t = patches_layer_padh<patches_layer_padh_desc<W_T, H_T, VS_T, HS_T, Filler_T, Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<weight_type_id>, Parameters...>::value,
        "Invalid parameters type for pooling_layer");
};

} //end of dll namespace

#endif
