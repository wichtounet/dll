//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_DESC_HPP
#define DLL_DBN_DESC_HPP

#include "base_dbn_desc.hpp"

namespace dll {

template<typename Layers, typename... Parameters>
using dbn_desc = base_dbn_desc<Layers, dbn, Parameters...>;

} //end of dll namespace

#endif