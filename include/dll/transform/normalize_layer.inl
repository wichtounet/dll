//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/data.hpp"

#include "transform_layer.hpp"

namespace dll {

/*!
 * \brief Simple thresholding normalize layer
 *
 * Note: This is only supported at the beginning of the network, no
 * backpropagation is possible for now.
 */
template <typename Desc>
struct normalize_layer : transform_layer<normalize_layer<Desc>> {
    using desc = Desc; ///< The descriptor type

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "Normalize";
    }

    /*!
     * \brief Apply the layer to the input
     * \param output The output
     * \param input The input to apply the layer to
     */
    template <typename Input, typename Output>
    static void activate_hidden(Output& output, const Input& input) {
        output = input;
        cpp::normalize(output);
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        output = input;
        cpp::normalize(output);
    }

    template<typename C>
    void adapt_errors(C& context) const {
        cpp_unused(context);
    }

    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        cpp_unused(output);
        cpp_unused(context);
    }

    template<typename C>
    void compute_gradients(C& context) const {
        cpp_unused(context);
    }
};

} //end of dll namespace
