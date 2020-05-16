//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

inline double gaussian(double x, double y, double sigma) {
    auto Z = 2.0 * M_PI * sigma * sigma;
    return (1.0 / Z) * std::exp(-((x * x + y * y) / (2.0 * sigma * sigma)));
}

template <typename W>
void lcn_filter(W& w, size_t K, size_t Mid, double sigma){
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < K; ++j) {
            w(i, j) = gaussian(double(i) - Mid, double(j) - Mid, sigma);
        }
    }

    w /= etl::sum(w);
}

/*!
 * \brief Apply the layer to the input
 * \param y The output
 * \param x The input to apply the layer to
 */
template <typename Input, typename Output, typename W>
void lcn_compute(Output&& y, const Input& x, const W& w, size_t K, size_t Mid){
    using weight_t = etl::value_t<Input>;

    auto v = etl::force_temporary(x(0));
    auto o = etl::force_temporary(x(0));

    for (size_t c = 0; c < etl::dim<0>(x); ++c) {
        //1. For each pixel, remove mean of 9x9 neighborhood
        for (size_t j = 0; j < etl::dim<1>(x); ++j) {
            for (size_t k = 0; k < etl::dim<2>(x); ++k) {
                weight_t sum(0.0);

                for (size_t p = 0; p < K; ++p) {
                    if (long(j) + p - Mid >= 0 && long(j) + p - Mid < etl::dim<1>(x)) {
                        for (size_t q = 0; q < K; ++q) {
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

        for (size_t j = 0; j < etl::dim<1>(x); ++j) {
            for (size_t k = 0; k < etl::dim<2>(x); ++k) {
                weight_t sum(0.0);

                for (size_t p = 0; p < K; ++p) {
                    if (long(j) + p - Mid >= 0 && long(j) + p - Mid < etl::dim<1>(x)) {
                        for (size_t q = 0; q < K; ++q) {
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

} //end of dll namespace
