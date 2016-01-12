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
using dbn_fast_desc = generic_dbn_desc<dbn_fast, Layers, Parameters...>;

} //end of dll namespace
