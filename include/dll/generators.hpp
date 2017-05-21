//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// Common helpers
#include "dll/generators/cache_helper.hpp"
#include "dll/generators/label_cache_helper.hpp"
#include "dll/generators/augmenters.hpp"

#include "dll/generators/inmemory_data_generator.hpp"

namespace dll {

template <typename T, typename = int>
struct is_generator : std::false_type {};

template <typename T>
struct is_generator<T, decltype((void)T::dll_generator, 0)> : std::true_type {};

} // end of namespace dll
