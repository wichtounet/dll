//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

//Stupid trick to allow auto functions to have debug symbols

#ifdef __clang__
#define CLANG_AUTO_TRICK template <typename E = void>
#else
#define CLANG_AUTO_TRICK
#endif
