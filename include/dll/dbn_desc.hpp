//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "generic_dbn_desc.hpp"

namespace dll {

template <typename Layers, typename... Parameters>
using dbn_desc = generic_dbn_desc<dbn, Layers, Parameters...>;

template <typename Layers>
struct dyn_layers_t;

template <bool Labels, typename... Layers>
struct dyn_layers_t <dll::detail::layers<Labels, Layers...>> {
    using dyn_t = dll::detail::layers<Labels, typename Layers::desc::dyn_layer_t...>;
};

template <template <typename> class DBN_T, typename Layers, typename... Parameters>
struct generic_dyn_dbn_desc : generic_dbn_desc<DBN_T, Layers, Parameters...> {
    /* Dynify the layers */
    using layers      = typename dyn_layers_t<Layers>::dyn_t;
    using base_layers = Layers;

    /*! The DBN type */
    using dbn_t = DBN_T<generic_dyn_dbn_desc<DBN_T, Layers, Parameters...>>;
};

template <typename Layers, typename... Parameters>
using dyn_dbn_desc = generic_dyn_dbn_desc<dbn, Layers, Parameters...>;

} //end of dll namespace
