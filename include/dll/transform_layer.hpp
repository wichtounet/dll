//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "neural_base.hpp"

namespace dll {

/*!
 * \brief Abstract transform layer
 *
 * Provide the base features for transform layer implementations.
 */
template <typename Derived>
struct transform_layer : neural_base<Derived> {
    using derived_t = Derived; ///< The derived type

    /*!
     * \brief Prints the layer to the console
     */
    static void display() {
        std::cout << derived_t::to_short_string() << std::endl;
    }

    /*!
     * \brief Apply the layer to many inputs
     * \param output The set of output
     * \param input The set of input to apply the layer to
     */
    template <typename I, typename O_A>
    static void activate_many(const I& input, O_A& output) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            derived_t::activate_hidden(input[i], output[i]);
        }
    }

    /*!
     * \brief Prepare a set of output
     * \param samples The number of samples in the output set
     */
    template <typename Input>
    static std::vector<Input> prepare_output(std::size_t samples) {
        return std::vector<Input>(samples);
    }

    /*!
     * \brief Prepare a single output
     */
    template <typename Input>
    static Input prepare_one_output() {
        return {};
    }
};

} //end of dll namespace
