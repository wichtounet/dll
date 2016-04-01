//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <typename Augment>
struct augmenter;

template <std::size_t C>
struct augmenter <copy<C>> {
    template <typename Input, typename Output>
    static void apply(Output& result, const Input& input) {
        for(std::size_t c = 0; c < C; ++c){
            // Simply create a copy
            result.push_back(input);
        }
    }

    static void concat_name(std::string& name) {
        name += " copy<" + std::to_string(C) + ">";
    }
};

template <std::size_t C, std::size_t K>
struct augmenter <elastic<C, K>> {
    static_assert(K % 2 == 1, "The kernel size must be odd");

    template<typename W>
    static void gaussian_blur(const etl::dyn_matrix<W>& d, etl::dyn_matrix<W>& d_blur){
        auto kernel_size = K;
        auto mid = kernel_size / 2;

        double sigma = 0.8 + 0.3 * ((kernel_size - 1) * 0.5 - 1);

        const std::size_t width = d.dim(0);
        const std::size_t height = d.dim(1);

        etl::dyn_matrix<W> kernel(kernel_size, kernel_size);

        auto gaussian = [](double x, double y, double sigma) {
            auto Z = 2.0 * M_PI * sigma * sigma;
            return (1.0 / Z) * std::exp(-((x * x + y * y) / (2.0 * sigma * sigma)));
        };

        for (std::size_t i = 0; i < kernel_size; ++i) {
            for (std::size_t j = 0; j < kernel_size; ++j) {
                kernel(i, j) = gaussian(double(i) - mid, double(j) - mid, sigma);;
            }
        }

        for (std::size_t j = 0; j < width; ++j) {
            for (std::size_t k = 0; k < height; ++k) {
                W sum(0.0);

                for (std::size_t p = 0; p < kernel_size; ++p) {
                    if (long(j) + p - mid >= 0 && long(j) + p - mid < width) {
                        for (std::size_t q = 0; q < kernel_size; ++q) {
                            if (long(k) + q - mid >= 0 && long(k) + q - mid < height) {
                                sum += kernel(p, q) * d(j + p - mid, k + q - mid);
                            }
                        }
                    }
                }

                d_blur(j, k) = d(j, k) - (sum / (kernel_size * kernel_size));
            }
        }
    }

    template <typename Input, typename Output>
    static void apply(Output& output, const Input& input) {
        for(std::size_t c = 0; c < C; ++c){
            // Create a copy of the same type
            auto result = input;

            using weight = etl::value_t<Input>;

            const std::size_t width = input.dim(1);
            const std::size_t height = input.dim(2);

            // 0. Generate random displacement fields

            etl::dyn_matrix<weight> d_x(width, height);
            etl::dyn_matrix<weight> d_y(width, height);

            d_x = etl::uniform_generator(-1.0, 1.0);
            d_y = etl::uniform_generator(-1.0, 1.0);

            // 1. Gaussian blur the displacement fields

            etl::dyn_matrix<weight> d_x_blur(width, height);
            etl::dyn_matrix<weight> d_y_blur(width, height);

            gaussian_blur(d_x, d_x_blur);
            gaussian_blur(d_y, d_y_blur);

            // 2. Normalize the displacement field

            d_x_blur /= sum(d_x_blur);
            d_y_blur /= sum(d_y_blur);

            // 3. Scale the displacement field

            weight alpha(8);

            d_x_blur *= alpha;
            d_y_blur *= alpha;

            // 4. Apply the displacement field (using bilinear interpolation)

            auto safe = [&](std::size_t channel, weight x, weight y) {
                if (x < 0 || y < 0 || x > width - 1 || y > height - 1) {
                    return input(channel, 0, 0);
                } else {
                    return input(channel, x, y);
                }
            };

            for (std::size_t channel = 0; channel < etl::dim<0>(input); ++channel) {
                for (int x = 0; x < int(width); ++x) {
                    for (int y = 0; y < int(height); ++y) {
                        auto dx = d_x_blur(x, y);
                        auto dy = d_y_blur(x, y);

                        weight px = x + dx;
                        weight py = y + dy;

                        weight a = safe(channel, std::floor(px), std::floor(py));
                        weight b = safe(channel, std::ceil(px), std::floor(py));
                        weight c = safe(channel, std::ceil(px), std::ceil(py));
                        weight d = safe(channel, std::floor(px), std::ceil(py));

                        auto e = a * (1.0 - (px - std::floor(px))) + d * (px - std::floor(px));
                        auto f = b * (1.0 - (px - std::floor(px))) + c * (px - std::floor(px));

                        auto value = e * (1.0 - (py - std::floor(py))) + f * (py - std::floor(py));

                        result(channel, x, y) = value;;
                    }
                }
            }

            output.push_back(result);
        }
    }

    static void concat_name(std::string& name) {
        name += " elastic<" + std::to_string(C) + ", " + std::to_string(K) + ">";
    }
};

} //end of dll namespace
