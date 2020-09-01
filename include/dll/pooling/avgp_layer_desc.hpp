//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "pooling_layer_desc.hpp"

namespace dll {

/*!
 * \brief Description of an Average Pooling two-dimensional layer.
 */
template <size_t T_I1, size_t T_I2, size_t T_I3, size_t T_C1, size_t T_C2, size_t T_S1, size_t T_S2, size_t T_P1, size_t T_P2, typename... Parameters>
struct avgp_2d_layer_desc : pooling_2d_layer_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_S1, T_S2, T_P1, T_P2, Parameters...> {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The layer type */
    using layer_t = avgp_2d_layer_impl<avgp_2d_layer_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_S1, T_S2, T_P1, T_P2, Parameters...>>;

    /*! The layer type */
    using dyn_layer_t = dyn_avgp_2d_layer_impl<dyn_avgp_2d_layer_desc<Parameters...>>;
};

/*!
 * \brief Description of an Average Pooling two-dimensional layer.
 */
template <size_t T_I1, size_t T_I2, size_t T_I3, size_t T_C1, size_t T_C2, typename... Parameters>
using avgp_2d_layer = typename avgp_2d_layer_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_C1, T_C2, 0, 0, Parameters...>::layer_t;

/*!
 * \brief Description of an Average Pooling two-dimensional layer.
 */
template <size_t T_I1, size_t T_I2, size_t T_I3, size_t T_C1, size_t T_C2, size_t T_S1, size_t T_S2, size_t T_P1, size_t T_P2, typename... Parameters>
using avgp_2d_layer_stride = typename avgp_2d_layer_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_S1, T_S2, T_P1, T_P2, Parameters...>::layer_t;

/*!
 * \brief Description of an Average Pooling three-dimensional layer.
 */
template <size_t T_I1, size_t T_I2, size_t T_I3, size_t T_C1, size_t T_C2, size_t T_C3, typename... Parameters>
struct avgp_3d_layer_desc : pooling_3d_layer_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_C3, Parameters...> {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*! The layer type */
    using layer_t = avgp_3d_layer_impl<avgp_3d_layer_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_C3, Parameters...>>;

    /*! The layer type */
    using dyn_layer_t = dyn_avgp_3d_layer_impl<dyn_avgp_3d_layer_desc<Parameters...>>;
};

/*!
 * \brief Description of an Average Pooling three-dimensional layer.
 */
template <size_t T_I1, size_t T_I2, size_t T_I3, size_t T_C1, size_t T_C2, size_t T_C3, typename... Parameters>
using avgp_3d_layer = typename avgp_3d_layer_desc<T_I1, T_I2, T_I3, T_C1, T_C2, T_C3, Parameters...>::layer_t;

} //end of dll namespace
