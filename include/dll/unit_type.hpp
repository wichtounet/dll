//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_UNIT_TYPE_HPP
#define DLL_UNIT_TYPE_HPP

namespace dll {

/*!
 * \brief A type of unit inside a RBM
 */
enum class unit_type {
    BINARY,     ///< Stochastic binary unity
    EXP,        ///< Exponential unit (for last layer)
    SOFTMAX,    ///< Softmax unit (for last layer)
    GAUSSIAN,   ///< Gaussian unit
    RELU,       ///< Rectified Linear Unit (ReLU) (Nair and Hinton, 2010)
    RELU1,      ///< Rectified Linear Unit (ReLU) capped at 1 (Krizhevsky, 2010)
    RELU6,      ///< Rectified Linear Unit (ReLU) capped at 6 (Krizhevsky,. 2010)
};

constexpr bool is_relu(unit_type t){
    return t == unit_type::RELU || t == unit_type::RELU1 || t == unit_type::RELU6;
}

} //end of dbn namespace

#endif