//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_ASSERT_HPP
#define DBN_ASSERT_HPP

#include <boost/assert.hpp>

#ifdef NDEBUG

#define dbn_assert(condition, message)
#define dbn_unreachable(message) __builtin_unreachable();

#else

#define dbn_assert(condition, message) BOOST_ASSERT_MSG(condition, message);

#ifdef __GNUC__

#define dbn_unreachable(message) BOOST_ASSERT_MSG(false, message); assert(false); __builtin_unreachable();

#endif //__GNUC__

#ifdef __clang__

#if __has_builtin(__builtin_unreachable)
#define dbn_unreachable(message) BOOST_ASSERT_MSG(false, message); assert(false); __builtin_unreachable();
#else
#define dbn_unreachable(message) BOOST_ASSERT_MSG(false, message); assert(false);
#endif

#endif //__clang__

#endif //NDEBUG

#endif //DBN_ASSERT_HPP
