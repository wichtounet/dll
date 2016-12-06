//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "transform_layer.hpp"
#include "lcn.hpp"

namespace dll {

/*!
 * \brief Local Contrast Normalization layer
 */
template <typename Desc>
struct dyn_lcn_layer : transform_layer<dyn_lcn_layer<Desc>> {
    using desc = Desc;

    size_t K;
    size_t Mid;
    double sigma = 2.0;

    void init_layer(size_t K){
        cpp_assert(K > 1, "The LCN kernel size must be greater than 1");
        cpp_assert(K % 2 == 1, "The LCN kernel size must be odd");

        this->K = K;
        this->Mid = K / 2;
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    std::string to_short_string() const {
        std::string desc("LCN(dyn): ");
        desc += std::to_string(K) + 'x' + std::to_string(K);
        return desc;
    }

    template <typename W>
    etl::dyn_matrix<W, 2> filter(double sigma) const {
        etl::dyn_matrix<W, 2> w(K, K);

        lcn_filter(w, K, Mid, sigma);

        return w;
    }

    /*!
     * \brief Apply the layer to the input
     * \param y The output
     * \param x The input to apply the layer to
     */
    template <typename Input, typename Output>
    void activate_hidden(Output& y, const Input& x) const {
        inherit_dim(y, x);

        using weight_t = etl::value_t<Input>;

        auto w = filter<weight_t>(sigma);

        lcn_compute(y, x, w, K, Mid);
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void batch_activate_hidden(Output& output, const Input& input) const {
        inherit_dim(output, input);

        for (std::size_t b = 0; b < etl::dim<0>(input); ++b) {
            activate_hidden(output(b), input(b));
        }
    }
};

} //end of dll namespace
