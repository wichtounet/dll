//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/util/random.hpp"

/*!
 * \brief Output policies for the network
 */

#pragma once

namespace dll {

/*!
 * \brief Default output policy
 */
struct default_output_policy {
    /*!
     * \brief Display the given value
     *
     * \param value The value to display
     */
    template <typename T>
    default_output_policy& operator<<(T&& value) {
        std::cout << std::forward<T>(value);

        return *this;
    }

    using manipulator = std::ostream& (std::ostream&);

    /*!
     * \brief Apply the given manipulator.
     *
     * \param m The manipulator to apply
     */
    default_output_policy& operator<<(manipulator& m) {
        std::cout << m;

        return *this;
    }
};

/*!
 * \brief Null output policy
 */
struct null_output_policy {
    using manipulator = std::ostream& (std::ostream&);

    /*!
     * \brief Display the given value
     *
     * \param value The value to display
     */
    template <typename T>
    null_output_policy& operator<<([[maybe_unused]] T&& value) {
        return *this;
    }

    /*!
     * \brief Apply the given manipulator.
     *
     * \param m The manipulator to apply
     */
    null_output_policy& operator<<([[maybe_unused]] manipulator& m) {
        return *this;
    }
};

} //end of dll namespace
