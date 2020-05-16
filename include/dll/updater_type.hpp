//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief The updater type for gradient descent
 */
enum class updater_type {
    SGD,          ///< The basic updater for SGD
    MOMENTUM,     ///< Use Momentum for SGD
    NESTEROV,     ///< Use Nesterov Momentum for SGD
    ADAGRAD,      ///< Use ADAGRAD for SGD
    RMSPROP,      ///< Use RMSPROP for SGD
    ADAM,         ///< Use Adam for SGD
    ADAM_CORRECT, ///< Use Adam with bias correction for SGD
    ADAMAX,       ///< Use Adamax for SGD
    NADAM,        ///< Use Nesterov Adam for SGD
    ADADELTA      ///< Use Adadelta for SGD
};

/*!
 * \brief Returns a string representation of an updater type
 * \param f The updater type to transform to string
 * \return a string representation of an updater type
 */
inline std::string to_string(updater_type f) {
    switch (f) {
        case updater_type::SGD:
            return "SGD";
        case updater_type::MOMENTUM:
            return "MOMENTUM";
        case updater_type::NESTEROV:
            return "NESTEROV";
        case updater_type::ADAGRAD:
            return "ADAGRAD";
        case updater_type::RMSPROP:
            return "RMSPROP";
        case updater_type::ADAM:
            return "ADAM";
        case updater_type::ADAM_CORRECT:
            return "ADAM_CORRECT";
        case updater_type::ADAMAX:
            return "ADAMAX";
        case updater_type::NADAM:
            return "NADAM";
        case updater_type::ADADELTA:
            return "ADADELTA";
    }

    cpp_unreachable("Unreachable code");

    return "UNDEFINED";
}

} //end of dll namespace
