//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "transform_layer.hpp"

namespace dll {

/*!
 * \brief Simple thresholding binarize layer
 *
 * Note: This is only supported at the beginning of the network, no
 * backpropagation is possible for now.
 */
template <typename Desc>
struct binarize_layer : transform_layer<binarize_layer<Desc>> {
    using desc = Desc; ///< The descriptor type

    static constexpr const std::size_t Threshold = desc::T;

    binarize_layer() = default;

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "Binarize";
    }

    /*!
     * \brief Apply the layer to the input
     * \param output The output
     * \param input The input to apply the layer to
     */
    template <typename Input, typename Output>
    static void activate_hidden(Output& output, const Input& input) {
        output = input;

        for (auto& value : output) {
            value = value > Threshold ? 1 : 0;
        }
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        output = input;

        for (auto& value : output) {
            value = value > Threshold ? 1 : 0;
        }
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

//Allow odr-use of the constexpr static members

template <typename Desc>
const std::size_t binarize_layer<Desc>::Threshold;

} //end of dll namespace
