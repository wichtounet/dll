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
 * \brief Configuraable rectifier layer.
 *
 * Use abs as a rectifier by default
 */
template <typename Desc>
struct rectifier_layer : transform_layer<rectifier_layer<Desc>> {
    using desc = Desc; ///< The descriptor type

    static constexpr const rectifier_method method = desc::method; ///< The rectifier method

    static_assert(method == rectifier_method::ABS, "Only ABS rectifier has been implemented");

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "Rectifier";
    }

    /*!
     * \brief Apply the layer to the input
     * \param output The output
     * \param input The input to apply the layer to
     */
    template <typename Input, typename Output>
    static void activate_hidden(Output& output, const Input& input) {
        if (method == rectifier_method::ABS) {
            output = etl::abs(input);
        }
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        if (method == rectifier_method::ABS) {
            output = etl::abs(input);
        }
    }
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const rectifier_method rectifier_layer<Desc>::method;

} //end of dll namespace
