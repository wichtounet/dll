//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
    NONE,         ///< No early stopping
    LOSS_GOAL,    ///< Stop early when goal loss is reached
    ERROR_GOAL,   ///< Stop early when goal error is reached
    LOSS_DIRECT,  ///< Stop early when loss is increasing
    ERROR_DIRECT, ///< Stop early when error is increasing
    LOSS_BEST,    ///< Stop early when loss is not going down the best
    ERROR_BEST    ///< Stop early when error is not going down the best
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
        case strategy::LOSS_DIRECT:
            return "Direct(loss)";
        case strategy::ERROR_DIRECT:
            return "Direct(error)";
        case strategy::LOSS_BEST:
            return "Best(loss)";
        case strategy::ERROR_BEST:
            return "Best(error)";
    }

    cpp_unreachable("Unreachable code");

    return "UNDEFINED";
}

/*!
 * \brief Indicates if the given strategy is based on error or loss
 * \param s The strategy to get information from
 * \return true if the strategy is based on error, false if it's based on loss.
 */
constexpr bool is_error(strategy s){
    return s == strategy::ERROR_GOAL || s == strategy::ERROR_DIRECT || s == strategy::ERROR_BEST;
}

} //end of dll namespace
