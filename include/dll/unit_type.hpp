//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief A type of unit inside a RBM
 */
enum class unit_type {
    BINARY,   ///< Stochastic binary unity
    SOFTMAX,  ///< Softmax unit (for last layer)
    GAUSSIAN, ///< Gaussian unit
    RELU,     ///< Rectified Linear Unit (ReLU) (Nair and Hinton, 2010)
    RELU1,    ///< Rectified Linear Unit (ReLU) capped at 1 (Krizhevsky, 2010)
    RELU6,    ///< Rectified Linear Unit (ReLU) capped at 6 (Krizhevsky,. 2010)
};

/*!
 * \brief Indicates if the given unit_type is a RELU type
 * \param t The type of unit to test
 * \return true if the given type is of a ReLU type
 */
constexpr bool is_relu(unit_type t) {
    return t == unit_type::RELU || t == unit_type::RELU1 || t == unit_type::RELU6;
}

/*!
 * \brief Return the string representation of a unit type
 * \param type The type to transform into string
 * \return the string representation of the unit type
 */
inline std::string to_string(unit_type type) {
    switch (type) {
        case unit_type::BINARY:
            return "BINARY";
        case unit_type::SOFTMAX:
            return "SOFTMAX";
        case unit_type::GAUSSIAN:
            return "GAUSSIAN";
        case unit_type::RELU:
            return "RELU";
        case unit_type::RELU1:
            return "RELU1";
        case unit_type::RELU6:
            return "RELU6";
    }

    cpp_unreachable("Unreachable code");

    return "UNDEFINED";
}

} //end of dll namespace
