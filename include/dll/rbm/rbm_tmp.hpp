//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/tmp.hpp"

#define H_PROBS(unit, ...) if constexpr (P && hidden_unit == unit) { __VA_ARGS__; }
#define H_PROBS2(hunit, vunit, ...) if constexpr (P && hidden_unit == hunit && visible_unit == vunit) { __VA_ARGS__; }
#define H_PROBS_MULTI(unit) if constexpr (P && hidden_unit == unit)
#define H_SAMPLE_INPUT(unit, ...) if constexpr (!P && S && hidden_unit == unit) { __VA_ARGS__; }
#define H_SAMPLE_INPUT_MULTI(unit) if constexpr (!P && S && hidden_unit == unit)
#define H_SAMPLE_PROBS(unit, ...) if constexpr (P && S && hidden_unit == unit) { __VA_ARGS__; }
#define H_SAMPLE_PROBS_MULTI(unit) if constexpr (P && S && hidden_unit == unit)

#define V_PROBS(unit, ...) if constexpr (P && visible_unit == unit) { __VA_ARGS__; }
#define V_PROBS_MULTI(unit) if constexpr (P && visible_unit == unit)
#define V_SAMPLE_INPUT(unit, ...) if constexpr (!P && S && visible_unit == unit) { __VA_ARGS__; }
#define V_SAMPLE_INPUT_MULTI(unit) if constexpr (!P && S && visible_unit == unit)
#define V_SAMPLE_PROBS(unit, ...) if constexpr (P && S && visible_unit == unit) { __VA_ARGS__; }
#define V_SAMPLE_PROBS_MULTI(unit) if constexpr (P && S && visible_unit == unit)
