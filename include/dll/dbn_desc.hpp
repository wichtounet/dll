//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "generic_dbn_desc.hpp"

namespace dll {

template <typename Layers>
struct dyn_layers_t;

template <bool Labels, typename... Layers>
struct dyn_layers_t <dll::detail::layers<Labels, Layers...>> {
    using dyn_t = dll::detail::layers<Labels, typename Layers::desc::dyn_layer_t...>;
};

template <template <typename> typename DBN_T, typename Layers, typename... Parameters>
struct generic_dyn_dbn_desc : generic_dbn_desc<DBN_T, Layers, Parameters...> {
    /* Dynify the layers */
    using layers      = typename dyn_layers_t<Layers>::dyn_t;
    using base_layers = Layers;

    /*! The DBN type */
    using dbn_t = DBN_T<generic_dyn_dbn_desc<DBN_T, Layers, Parameters...>>;

    /*!
     * \brief The network type.
     *
     * This is the same as the DBN type, only kept for legacy
     * reasons.
     */
    using network_t = dbn_t;
};

/*!
 * \brief A descriptor for a multi-layer dynamic network.
 * \tparam Layers The set of layers
 * \tparam Parameters The set of parameters for this network
 */
template <typename Layers, typename... Parameters>
using dyn_dbn_desc = generic_dyn_dbn_desc<dbn, Layers, Parameters...>;

/*!
 * \brief A descriptor for a multi-layer dynamic network.
 * \tparam Layers The set of layers
 * \tparam Parameters The set of parameters for this network
 */
template <typename Layers, typename... Parameters>
using dyn_network_desc = generic_dyn_dbn_desc<dbn, Layers, Parameters...>;

// By default dbn_desc use directly the layers that it is provided
// if DLL_QUICK is set, it is set to hybrid mode by default

#ifndef DLL_QUICK

/*!
 * \brief A descriptor for a multi-layer network.
 *
 * If DLL_QUICK if set, this will default to use dynamic layers
 * instead of the provided layers when possible. Otherwise, the
 * layers will be used as is.
 *
 * \tparam Layers The set of layers \tparam Parameters The set of
 * parameters for this network
 */
template <typename Layers, typename... Parameters>
using dbn_desc = generic_dbn_desc<dbn, Layers, Parameters...>;

/*!
 * \brief A descriptor for a multi-layer network.
 *
 * If DLL_QUICK if set, this will default to use dynamic layers
 * instead of the provided layers when possible. Otherwise, the
 * layers will be used as is.
 *
 * \tparam Layers The set of layers \tparam Parameters The set of
 * parameters for this network
 */
template <typename Layers, typename... Parameters>
using network_desc = generic_dbn_desc<dbn, Layers, Parameters...>;

#else

/*!
 * \brief A descriptor for a multi-layer network.
 *
 * If DLL_QUICK if set, this will default to use dynamic layers
 * instead of the provided layers when possible. Otherwise, the
 * layers will be used as is.
 *
 * \tparam Layers The set of layers \tparam Parameters The set of
 * parameters for this network
 */
template <typename Layers, typename... Parameters>
using dbn_desc = generic_dyn_dbn_desc<dbn, Layers, Parameters...>;

/*!
 * \brief A descriptor for a multi-layer network.
 *
 * If DLL_QUICK if set, this will default to use dynamic layers
 * instead of the provided layers when possible. Otherwise, the
 * layers will be used as is.
 *
 * \tparam Layers The set of layers \tparam Parameters The set of
 * parameters for this network
 */
template <typename Layers, typename... Parameters>
using network_desc = generic_dyn_dbn_desc<dbn, Layers, Parameters...>;

#endif

// fast_dbn_desc is always forced to direct mode, will not respect
// DLL_QUICK

/*!
 * \brief A descriptor for a multi-layer network.
 *
 * This descriptor will always use the layers it is provided as is.
 *
 * \tparam Layers The set of layers
 * \tparam Parameters The set of parameters for this network
 */
template <typename Layers, typename... Parameters>
using fast_dbn_desc = generic_dbn_desc<dbn, Layers, Parameters...>;

/*!
 * \brief A descriptor for a multi-layer network.
 *
 * This descriptor will always use the layers it is provided as is.
 *
 * \tparam Layers The set of layers
 * \tparam Parameters The set of parameters for this network
 */
template <typename Layers, typename... Parameters>
using fast_network_desc = generic_dbn_desc<dbn, Layers, Parameters...>;

} //end of dll namespace
