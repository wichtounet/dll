//=======================================================================
// Copyright Baptiste Wicht 2014.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#ifndef DBN_ASSERT_HPP
#define DBN_ASSERT_HPP

#include <cassert>

#include <boost/assert.hpp>

#define dbn_assert(condition, message) BOOST_ASSERT_MSG(condition, message);

#ifdef __GNUC__

#define dbn_unreachable(message) BOOST_ASSERT_MSG(false, message); assert(false); __builtin_unreachable();

#endif

#ifdef __clang__

#if __has_builtin(__builtin_unreachable)
#define dbn_unreachable(message) BOOST_ASSERT_MSG(false, message); assert(false); __builtin_unreachable();
#else
#define dbn_unreachable(message) BOOST_ASSERT_MSG(false, message); assert(false);
#endif

#endif

#endif
