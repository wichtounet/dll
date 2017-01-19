//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/layer.hpp"

namespace dll {

/*!
 * \brief Abstract transform layer
 *
 * Provide the base features for transform layer implementations.
 */
template <typename Derived>
struct transform_layer : layer<Derived> {
    using derived_t = Derived; ///< The derived type

    transform_layer() = default;

    /*!
     * \brief Apply the layer to many inputs
     * \param output The set of output
     * \param input The set of input to apply the layer to
     */
    template <typename I, typename O_A>
    void activate_many(const I& input, O_A& output) const {
        for (std::size_t i = 0; i < input.size(); ++i) {
            as_derived().activate_hidden(input[i], output[i]);
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

    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
    }

private:
    const derived_t& as_derived() const {
        return *static_cast<const derived_t*>(this);
    }
};

/*!
 * \brief Make the output inherit the dimensions of the input, if
 * necessary
 * \param output The output
 * \param input The input
 */
template <typename Input, typename Output, cpp_enable_if(etl::all_fast<Output>::value)>
void inherit_dim(Output& output, const Input& input) {
    cpp_unused(output);
    cpp_unused(input);
    //Nothing to do, the output is fast (fixed dimensions)
}

/*!
 * \brief Make the output inherit the dimensions of the input, if
 * necessary
 * \param output The output
 * \param input The input
 */
template <typename Input, typename Output, cpp_disable_if(etl::all_fast<Output>::value)>
void inherit_dim(Output& output, const Input& input) {
    output.inherit_if_null(input);
}

} //end of dll namespace
