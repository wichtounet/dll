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

namespace dll {

template <typename T, typename = int>
struct is_generator : std::false_type {};

template <typename T>
struct is_generator<T, decltype((void)T::dll_generator, 0)> : std::true_type {};

template<typename Desc>
struct is_augmented {
    static constexpr bool value = (Desc::random_crop_x > 0 && Desc::random_crop_y > 0) || Desc::HorizontalMirroring || Desc::VerticalMirroring || Desc::Noise || Desc::ElasticDistortion;
};

template<typename Desc>
struct is_threaded {
    static constexpr bool value = Desc::Threaded;
};

} // end of namespace dll

#include "dll/generators/inmemory_data_generator.hpp"
#include "dll/generators/outmemory_data_generator.hpp"
