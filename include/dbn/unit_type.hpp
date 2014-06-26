//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_UNIT_TYPE_HPP
#define DBN_UNIT_TYPE_HPP

namespace dbn {

enum class Type {
    SIGMOID,    //Stochastic binary unity
    EXP,        //Exponential unit (for last layer)
    SOFTMAX,    //Softmax unit (for last layer)
    GAUSSIAN,   //Gaussian visible layers
    NRLU        //Noisy Rectified Linear Unit (nRLU)
};

} //end of dbn namespace

#endif