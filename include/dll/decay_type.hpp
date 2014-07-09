//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_DECAY_TYPE_HPP
#define DBN_DECAY_TYPE_HPP

namespace dll {

enum class decay_type {
    NONE,
    L1,
    L2,
    L1_FULL,
    L2_FULL
};

} //end of dbn namespace

#endif