//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifndef NAN_DEBUG

#define nan_check(value) ((void)0)
#define nan_check_etl(value) ((void)0)
#define nan_check_deep(list) ((void)0)
#define nan_check_deep_deep(list) ((void)0)
#define nan_check_deep_3(l1, l2, l3) ((void)0)

#else

#include <cmath> //for std::isfinite

#include "cpp_utils/assert.hpp"

#define nan_check(value) cpp_assert(std::isfinite(((value))), "NaN Verify");
#define nan_check_etl(value) cpp_assert(((value)).is_finite(), "NaN Verify");
#define nan_check_deep(list)                           \
    for (auto& _nan : ((list))) {                      \
        cpp_assert(std::isfinite(_nan), "NaN Verify"); \
    }
#define nan_check_deep_deep(l)                               \
    for (auto& _nan_a : ((l))) {                             \
        for (auto& _nan_b : _nan_a) {                        \
            cpp_assert(std::isfinite(_nan_b), "NaN Verify"); \
        }                                                    \
    }
#define nan_check_deep_3(l1, l2, l3) \
    nan_check_deep(l1);              \
    nan_check_deep(l2);              \
    nan_check_deep(l3);

#endif //NDEBUG
