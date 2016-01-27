//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_AVGP_LAYER_DESC_HPP
#define DLL_AVGP_LAYER_DESC_HPP

#include "pooling_layer_desc.hpp"

namespace dll {

template <std::size_t T_I1, std::size_t T_I2, std::size_t T_I3, std::size_t T_C1, std::size_t T_C2, std::size_t T_C3, typename... Parameters>
struct avgp_layer_3d_desc : pooling_layer_3d_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_C3, Parameters...> {
    using parameters = cpp::type_list<Parameters...>;

    /*! The layer type */
    using layer_t = avgp_layer_3d<avgp_layer_3d_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_C3, Parameters...>>;
};

} //end of dll namespace

#endif
