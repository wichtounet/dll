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

    static cpp14_constexpr double gaussian(double x, double y, double sigma) {
        auto Z = 2.0 * M_PI * sigma * sigma;
        return (1.0 / Z) * std::exp(-((x * x + y * y) / (2.0 * sigma * sigma)));
    }

    template <typename W>
    static etl::fast_dyn_matrix<W, K, K> filter(double sigma) {
        etl::fast_dyn_matrix<W, K, K> w;

        for (std::size_t i = 0; i < K; ++i) {
            for (std::size_t j = 0; j < K; ++j) {
                w(i, j) = gaussian(double(i) - Mid, double(j) - Mid, sigma);
            }
        }

        w /= etl::sum(w);

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

        auto v = etl::force_temporary(x(0));
        auto o = etl::force_temporary(x(0));

        for (std::size_t c = 0; c < etl::dim<0>(x); ++c) {
            //1. For each pixel, remove mean of 9x9 neighborhood
            for (std::size_t j = 0; j < etl::dim<1>(x); ++j) {
                for (std::size_t k = 0; k < etl::dim<2>(x); ++k) {
                    weight_t sum(0.0);

                    for (std::size_t p = 0; p < K; ++p) {
                        if (long(j) + p - Mid >= 0 && long(j) + p - Mid < etl::dim<1>(x)) {
                            for (std::size_t q = 0; q < K; ++q) {
                                if (long(k) + q - Mid >= 0 && long(k) + q - Mid < etl::dim<2>(x)) {
                                    sum += w(p, q) * x(c, j + p - Mid, k + q - Mid);
                                }
                            }
                        }
                    }

                    v(j, k) = x(c, j, k) - sum;
                }
            }

            //2. Scale down norm of 9x9 patch if norm is bigger than 1

            for (std::size_t j = 0; j < etl::dim<1>(x); ++j) {
                for (std::size_t k = 0; k < etl::dim<2>(x); ++k) {
                    weight_t sum(0.0);

                    for (std::size_t p = 0; p < K; ++p) {
                        if (long(j) + p - Mid >= 0 && long(j) + p - Mid < etl::dim<1>(x)) {
                            for (std::size_t q = 0; q < K; ++q) {
                                if (long(k) + q - Mid >= 0 && long(k) + q - Mid < etl::dim<2>(x)) {
                                    sum += w(p, q) * x(c, j + p - Mid, k + q - Mid) * x(c, j + p - Mid, k + q - Mid);
                                }
                            }
                        }
                    }

                    o(j, k) = std::sqrt(sum);
                }
            }

            auto cst = etl::mean(o);
            y(c) = v / etl::max(o, cst);
        }
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
