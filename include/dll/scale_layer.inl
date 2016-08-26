//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "transform_layer.hpp"

namespace dll {

//TODO This is not as generic as it should be
//It only supports fast_matrix input otherwise the size of input and
//output will be different and it will throw an assert One easy
//solution would be to extend the support for empty matrix in ETL

/*!
 * \brief Simple scaling layer
 */
template <typename Desc>
struct scale_layer : transform_layer<scale_layer<Desc>> {
    using desc = Desc; ///< The descriptor type

    static constexpr const int A = desc::A; ///< The scale multiplier
    static constexpr const int B = desc::B; ///< The scale divisor

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "scale";
    }

    /*!
     * \brief Apply the layer to the input
     * \param output The output
     * \param input The input to apply the layer to
     */
    template <typename Input, typename Output>
    static void activate_hidden(Output& output, const Input& input) {
        output = input * (double(A) / double(B));
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        output = input * (double(A) / double(B));
    }

    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
    }
};

} //end of dll namespace
