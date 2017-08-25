//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/etl.hpp"

#include "dll/layer_traits.hpp"

namespace dll {

/*!
 * \brief Prepare a ready output for the given layer from the given input.
 *
 * A ready output as all its dimensions set correctly.
 *
 * \param layer The layer to use to generate the output
 * \param input The input to the layer
 *
 * \return The all-ready output
 */
template<typename Layer, typename Input, cpp_enable_if(decay_layer_traits<Layer>::is_transform_layer() && !etl::all_fast<Input>)>
auto prepare_one_ready_output(Layer& layer, const Input& input){
    auto out = layer.template prepare_one_output<Input>();

    // At this point, the dimensions are not ready, so inherit
    out.inherit_if_null(input);

    return out;
}

/*!
 * \brief Prepare a ready output for the given layer from the given input.
 *
 * A ready output as all its dimensions set correctly.
 *
 * \param layer The layer to use to generate the output
 * \param input The input to the layer
 *
 * \return The all-ready output
 */
template<typename Layer, typename Input, cpp_enable_if(decay_layer_traits<Layer>::is_transform_layer() && etl::all_fast<Input>)>
auto prepare_one_ready_output(Layer& layer, const Input& input){
    cpp_unused(input);

    // A transform layer using fast input does not need inherit
    return layer.template prepare_one_output<Input>();
}

/*!
 * \brief Prepare a ready output for the given layer from the given input.
 *
 * A ready output as all its dimensions set correctly.
 *
 * \param layer The layer to use to generate the output
 * \param input The input to the layer
 *
 * \return The all-ready output
 */
template<typename Layer, typename Input, cpp_disable_if(decay_layer_traits<Layer>::is_transform_layer())>
auto prepare_one_ready_output(Layer& layer, const Input& input){
    cpp_unused(input);

    return layer.template prepare_one_output<Input>();
}

/*!
 * \brief Prepare a collection of ready output for the given layer from the given input.
 *
 * A ready output as all its dimensions set correctly.
 *
 * \param layer The layer to use to generate the output
 * \param input The input to the layer
 * \param n The number of samples to prepare
 *
 * \return The collection of all-ready output
 */
template<typename Layer, typename Input, cpp_enable_if(decay_layer_traits<Layer>::is_transform_layer() && !etl::all_fast<Input>)>
auto prepare_many_ready_output(Layer& layer, const Input& input, size_t n){
    auto out = layer.template prepare_output<Input>(n);

    // At this point, the dimensions are not ready, so inherit
    for (auto& x : out) {
        x.inherit_if_null(input);
    }

    return out;
}

/*!
 * \brief Prepare a collection of ready output for the given layer from the given input.
 *
 * A ready output as all its dimensions set correctly.
 *
 * \param layer The layer to use to generate the output
 * \param input The input to the layer
 * \param n The number of samples to prepare
 *
 * \return The collection of all-ready output
 */
template<typename Layer, typename Input, cpp_enable_if(decay_layer_traits<Layer>::is_transform_layer() && etl::all_fast<Input>)>
auto prepare_many_ready_output(Layer& layer, const Input& input, size_t n){
    cpp_unused(input);

    // A transform layer using fast input does not need inherit
    return layer.template prepare_output<Input>(n);
}

/*!
 * \brief Prepare a collection of ready output for the given layer from the given input.
 *
 * A ready output as all its dimensions set correctly.
 *
 * \param layer The layer to use to generate the output
 * \param input The input to the layer
 * \param n The number of samples to prepare
 *
 * \return The collection of all-ready output
 */
template<typename Layer, typename Input, cpp_disable_if(decay_layer_traits<Layer>::is_transform_layer())>
auto prepare_many_ready_output(Layer& layer, const Input& input, size_t n){
    cpp_unused(input);

    return layer.template prepare_output<Input>(n);
}

} //end of dll namespace
