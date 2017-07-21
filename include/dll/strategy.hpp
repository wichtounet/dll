//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief The strategy for early stopping
 */
enum class strategy {
    NONE,      ///< No early stopping
    LOSS_GOAL, ///< Stop early when goal loss is reached
    ERROR_GOAL ///< Stop early when goal error is reached
};

/*!
 * \brief Returns a string representation of a strategy type
 * \param f The strategy type to transform to string
 * \return a string representation of a strategy type
 */
inline std::string to_string(strategy s) {
    switch (s) {
        case strategy::NONE:
            return "None";
        case strategy::LOSS_GOAL:
            return "Goal(loss)";
        case strategy::ERROR_GOAL:
            return "Goal(error)";
    }

    cpp_unreachable("Unreachable code");

    return "UNDEFINED";
}

} //end of dll namespace
