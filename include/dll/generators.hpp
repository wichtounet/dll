//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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
 * \brief Traits to test if a type is DLL generator or not
 */
template <typename T>
concept generator = is_generator_impl<T>::value;

/*!
 * \brief Traits to test if a type is DLL generator or not
 */
template <typename T>
concept not_generator = !generator<T>;

template <typename Desc>
concept augmented_generator = (Desc::random_crop_x > 0 && Desc::random_crop_y > 0) || Desc::HorizontalMirroring || Desc::VerticalMirroring || Desc::Noise != 0
                              || Desc::ElasticDistortion != 0;

template<typename Desc>
concept threaded_generator = Desc::Threaded;

template<typename Desc>
concept standard_generator = !augmented_generator<Desc> && !threaded_generator<Desc>;

template<typename Desc>
concept special_generator = augmented_generator<Desc> || threaded_generator<Desc>;

} // end of namespace dll

#include "dll/generators/inmemory_data_generator.hpp"
#include "dll/generators/inmemory_single_data_generator.hpp"
#include "dll/generators/outmemory_data_generator.hpp"
