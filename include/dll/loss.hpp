//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file loss.hpp
 * \brief Losses for neural network training
 */

#pragma once

namespace dll {

/*!
 * \brief An activation function
 */
enum class loss_function {
    CATEGORICAL_CROSS_ENTROPY, ///< Categorical Cross Entropy Loss
    BINARY_CROSS_ENTROPY,      ///< Binary Cross Entropy Loss
    MEAN_SQUARED_ERROR         ///< Mean Squared Error Loss
};

/*!
 * \brief Returns a string representation of a loss function
 * \param f The loss to transform to string
 * \return a string representation of a loss function
 */
inline std::string to_string(loss_function f) {
    switch (f) {
        case loss_function::CATEGORICAL_CROSS_ENTROPY:
            return "CATEGORICAL_CROSS_ENTROPY";
        case loss_function::BINARY_CROSS_ENTROPY:
            return "BINARY_CROSS_ENTROPY";
        case loss_function::MEAN_SQUARED_ERROR:
            return "MEAN_SQUARED_ERROR";
    }

    cpp_unreachable("Unreachable code");

    return "UNDEFINED";
}

} //end of dll namespace
