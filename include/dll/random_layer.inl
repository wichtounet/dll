//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "transform_layer.hpp"

namespace dll {

/*!
 * \brief Test layer that generate random outputs
 */
template <typename Desc>
struct random_layer : transform_layer<random_layer<Desc>> {
    using desc = Desc; ///< The descriptor type

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "Random";
    }

    /*!
     * \brief Apply the layer to the input
     * \param output The output
     * \param input The input to apply the layer to
     */
    template <typename Input, typename Output>
    static void activate_hidden(Output& output, const Input& input) {
        inherit_dim(output, input);
        output = etl::normal_generator<etl::value_t<Input>>();
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        inherit_dim(output, input);
        output = etl::normal_generator<etl::value_t<Input>>();
    }
};

} //end of dll namespace
