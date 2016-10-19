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
struct lcn_layer : transform_layer<lcn_layer<Desc>> {
    using desc = Desc; ///< The descriptor type

    static constexpr const std::size_t K = desc::K;
    static constexpr const std::size_t Mid = K / 2;

    double sigma = 2.0;

    static_assert(K > 1, "The kernel size must be greater than 1");
    static_assert(K % 2 == 1, "The kernel size must be odd");

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        std::string desc("LCN: ");
        desc += std::to_string(K) + 'x' + std::to_string(K);
        return desc;
    }

    template <typename W>
    static etl::fast_dyn_matrix<W, K, K> filter(double sigma) {
        etl::fast_dyn_matrix<W, K, K> w;

        lcn_filter(w, K, Mid, sigma);

        return w;
    }

    /*!
     * \brief Apply the layer to the input
     * \param output The output
     * \param input The input to apply the layer to
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

    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(K);
    }
};

} //end of dll namespace
