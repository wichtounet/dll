//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_ASSERT_HPP
#define DBN_ASSERT_HPP

#include <iostream>

#include "likely.hpp"

#ifdef NDEBUG

#define nan_check(list) ((void)0)
#define nan_check_3(l1, l2, l3) ((void)0)

#define dll_assert(condition, message) ((void)0)

#if defined __clang__

#if __has_builtin(__builtin_unreachable)
#define dbn_unreachable(message) __builtin_unreachable();
#else
#define dbn_unreachable(message) ((void)0)
#endif //__has_builtin(__builtin_unreachable)

#elif defined __GNUC__

#define dbn_unreachable(message) __builtin_unreachable();

#endif //__clang__

#else

#define nan_check(list) for(auto& nantest : ((list))){dll_assert(std::isfinite(nantest), "NaN Verify");}
#define nan_check_3(l1,l2,l3) nan_check(l1); nan_check(l2); nan_check(l3);

#define dll_assert(condition, message) (likely(condition) \
    ? ((void)0) \
    : ::dll::assertion::detail::assertion_failed_msg(#condition, message, \
    __PRETTY_FUNCTION__, __FILE__, __LINE__))

#if defined __clang__

#if __has_builtin(__builtin_unreachable)
#define dbn_unreachable(message) dll_assert(false, message); __builtin_unreachable();
#else
#define dbn_unreachable(message) dll_assert(false, message);
#endif //__has_builtin(__builtin_unreachable)

#elif defined __GNUC__

#define dbn_unreachable(message) dll_assert(false, message); __builtin_unreachable();

#endif //__clang__

#endif //NDEBUG

namespace dll {
namespace assertion {
namespace detail {

    template< typename CharT >
    void assertion_failed_msg(const CharT* expr, const char* msg, const char* function, const char* file, long line){
        std::cerr
            << "***** Internal Program Error - assertion (" << expr << ") failed in "
            << function << ":\n"
            << file << '(' << line << "): " << msg << std::endl;
        std::abort();
    }

} // end of detail namespace
} // end of assertion namespace
} // end of detail namespace

#endif //DBN_ASSERT_HPP
