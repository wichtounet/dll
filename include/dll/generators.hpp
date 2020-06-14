//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/etl.hpp"

#include "dll/util/tmp.hpp"
#include "dll/base_conf.hpp"

// Common helpers
#include "dll/generators/cache_helper.hpp"
#include "dll/generators/label_cache_helper.hpp"
#include "dll/generators/augmenters.hpp"
#include "dll/generators/transformers.hpp"

namespace dll {

/*!
 * \brief Traits to test if a type is DLL generator or not
 */
template <typename T, typename = int>
struct is_generator_impl : std::false_type {};

/*!
 * \brief Traits to test if a type is DLL generator or not
 */
template <typename T>
struct is_generator_impl<T, decltype((void)T::dll_generator, 0)> : std::true_type {};

/*!
 * \brief Traits to test if a type is DLL generator or not
 */
template <typename T>
constexpr bool is_generator = is_generator_impl<T>::value;

/*!
 * \brief Helper to tell from the generator description if it is
 * augmenting the data
 */
template<typename Desc>
constexpr bool is_augmented =
            (Desc::random_crop_x > 0 && Desc::random_crop_y > 0)
        ||  Desc::HorizontalMirroring || Desc::VerticalMirroring || Desc::Noise || Desc::ElasticDistortion;

/*!
 * \brief Helper to tell from the generator description if it is
 * threaded.
 */
template<typename Desc>
static constexpr bool is_threaded = Desc::Threaded;

} // end of namespace dll

#include "dll/generators/inmemory_data_generator.hpp"
#include "dll/generators/inmemory_single_data_generator.hpp"
#include "dll/generators/outmemory_data_generator.hpp"
