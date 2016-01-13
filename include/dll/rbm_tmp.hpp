//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/tmp.hpp"
#include "cpp_utils/static_if.hpp"

#define H_PROBS(unit, ...) cpp::static_if<P && hidden_unit == unit>([&](auto f){ __VA_ARGS__ ; });
#define H_PROBS2(hunit, vunit, ...) cpp::static_if<P && hidden_unit == hunit && visible_unit == vunit>([&](auto f){ __VA_ARGS__ ; });
#define H_PROBS_MULTI(unit) cpp::static_if<P && hidden_unit == unit>
#define H_SAMPLE_INPUT(unit, ...) cpp::static_if<!P && S && hidden_unit == unit>([&](auto f){ __VA_ARGS__ ; });
#define H_SAMPLE_INPUT_MULTI(unit) cpp::static_if<!P && S && hidden_unit == unit>
#define H_SAMPLE_PROBS(unit, ...) cpp::static_if<P && S && hidden_unit == unit>([&](auto f){ __VA_ARGS__ ; });
#define H_SAMPLE_PROBS_MULTI(unit) cpp::static_if<P && S && hidden_unit == unit>

#define V_PROBS(unit, ...) cpp::static_if<P && visible_unit == unit>([&](auto f){ __VA_ARGS__ ; });
#define V_PROBS_MULTI(unit) cpp::static_if<P && visible_unit == unit>
#define V_SAMPLE_INPUT(unit, ...) cpp::static_if<!P && S && visible_unit == unit>([&](auto f){ __VA_ARGS__ ; });
#define V_SAMPLE_INPUT_MULTI(unit) cpp::static_if<!P && S && visible_unit == unit>
#define V_SAMPLE_PROBS(unit, ...) cpp::static_if<P && S && visible_unit == unit>([&](auto f){ __VA_ARGS__ ; });
#define V_SAMPLE_PROBS_MULTI(unit) cpp::static_if<P && S && visible_unit == unit>
