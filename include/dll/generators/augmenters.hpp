//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <atomic>
#include <thread>

namespace dll {

template<typename Desc, typename Enable = void>
struct random_cropper;

template<typename Desc>
struct random_cropper <Desc, std::enable_if_t<Desc::random_crop_x && Desc::random_crop_y>> {
    using weight = typename Desc::weight;

    static constexpr size_t random_crop_x = Desc::random_crop_x;
    static constexpr size_t random_crop_y = Desc::random_crop_y;

    size_t x = 0;
    size_t y = 0;

    std::random_device rd;
    std::default_random_engine engine;

    std::uniform_int_distribution<size_t> dist_x;
    std::uniform_int_distribution<size_t> dist_y;

    template<typename T>
    random_cropper(const T& image) : engine(rd()) {
        static_assert(etl::dimensions<T>() == 3, "random_cropper can only be used with 3D images");

        y = etl::dim<1>(image);
        x = etl::dim<2>(image);

        dist_x = std::uniform_int_distribution<size_t>{0, x - random_crop_x};
        dist_y = std::uniform_int_distribution<size_t>{0, y - random_crop_y};
    }

    size_t scaling() const {
        return (x - random_crop_x) * (y - random_crop_y);
    }

    template<typename O, typename T>
    void transform_first(O&& target, const T& image){
        const size_t y_offset = dist_y(engine);
        const size_t x_offset = dist_x(engine);

        for (size_t c = 0; c < etl::dim<0>(image); ++c) {
            for (size_t y = 0; y < random_crop_y; ++y) {
                for (size_t x = 0; x < random_crop_x; ++x) {
                    target(c, y, x) = image(c, y_offset + y, x_offset + x);
                }
            }
        }
    }

    template<typename O, typename T>
    void transform_first_test(O&& target, const T& image){
        const size_t y_offset = (x - random_crop_x) / 2;
        const size_t x_offset = (y - random_crop_y) / 2;

        for (size_t c = 0; c < etl::dim<0>(image); ++c) {
            for (size_t y = 0; y < random_crop_y; ++y) {
                for (size_t x = 0; x < random_crop_x; ++x) {
                    target(c, y, x) = image(c, y_offset + y, x_offset + x);
                }
            }
        }
    }
};

template<typename Desc>
struct random_cropper <Desc, std::enable_if_t<!Desc::random_crop_x || !Desc::random_crop_y>> {
    template<typename T>
    random_cropper(const T& image){
        cpp_unused(image);
    }

    static constexpr size_t scaling() {
        return 1;
    }

    template<typename O, typename T>
    void transform_first(O&& target, const T& image){
        target = image;
    }

    template<typename O, typename T>
    void transform_first_test(O&& target, const T& image){
        target = image;
    }
};

template<typename Desc, typename Enable = void>
struct random_mirrorer;

template<typename Desc>
struct random_mirrorer <Desc, std::enable_if_t<Desc::HorizontalMirroring || Desc::VerticalMirroring>> {
    static constexpr bool horizontal = Desc::HorizontalMirroring;
    static constexpr bool vertical = Desc::VerticalMirroring;

    std::random_device rd;
    std::default_random_engine engine;

    std::uniform_int_distribution<size_t> dist;

    template<typename T>
    random_mirrorer(const T& image) : engine(rd()) {
        static_assert(etl::dimensions<T>() == 3, "random_mirrorer can only be used with 3D images");

        if(horizontal && vertical){
            dist = std::uniform_int_distribution<size_t>{0, 3};
        } else {
            dist = std::uniform_int_distribution<size_t>{0, 2};
        }

        cpp_unused(image);
    }

    size_t scaling() const {
        if(horizontal && vertical){
            return 3;
        } else {
            return 2;
        }
    }

    template<typename O>
    void transform(O&& target){
        auto choice = dist(engine);

        if(horizontal && vertical && choice == 1){
            for(size_t c= 0; c < etl::dim<0>(target); ++c){
                target(c) = vflip(target(c));
            }
        } else if(horizontal && vertical && choice == 2){
            for(size_t c= 0; c < etl::dim<0>(target); ++c){
                target(c) = hflip(target(c));
            }
        }

        if(horizontal && choice == 1){
            for(size_t c= 0; c < etl::dim<0>(target); ++c){
                target(c) = hflip(target(c));
            }
        }

        if(vertical && choice == 1){
            for(size_t c= 0; c < etl::dim<0>(target); ++c){
                target(c) = vflip(target(c));
            }
        }
    }
};

template<typename Desc>
struct random_mirrorer <Desc, std::enable_if_t<!Desc::HorizontalMirroring && !Desc::VerticalMirroring>> {
    template<typename T>
    random_mirrorer(const T& image){
        cpp_unused(image);
    }

    static constexpr size_t scaling() {
        return 1;
    }

    template<typename O>
    static void transform(O&& target){
        cpp_unused(target);
    }
};

template<typename Desc, typename Enable = void>
struct random_noise;

template<typename Desc>
struct random_noise <Desc, std::enable_if_t<Desc::Noise>> {
    using weight = typename Desc::weight;

    static constexpr size_t N = Desc::Noise;

    std::random_device rd;
    std::default_random_engine engine;

    std::uniform_int_distribution<size_t> dist;

    template<typename T>
    random_noise(const T& image) : engine(rd()), dist(0, 1000) {
        cpp_unused(image);
    }

    size_t scaling() const {
        return 10;
    }

    template<typename O>
    void transform(O&& target){
        for(auto& v :  target){
            v *= dist(engine) < N * 10 ? 0.0 : 1.0;
        }
    }
};

template<typename Desc>
struct random_noise <Desc, std::enable_if_t<!Desc::Noise>> {
    template<typename T>
    random_noise(const T& image){
        cpp_unused(image);
    }

    static constexpr size_t scaling() {
        return 1;
    }

    template<typename O>
    static void transform(O&& target){
        cpp_unused(target);
    }
};

template<typename Desc, typename Enable = void>
struct elastic_distorter;

// TODO This needs to be made MUCH faster

template<typename Desc>
struct elastic_distorter <Desc, std::enable_if_t<Desc::ElasticDistortion>> {
    using weight = typename Desc::weight;

    static constexpr size_t K     = Desc::ElasticDistortion;
    static constexpr size_t mid   = K / 2;
    static constexpr double sigma = 0.8 + 0.3 * ((K - 1) * 0.5 - 1);

    etl::fast_dyn_matrix<weight, K, K> kernel;

    static_assert(K % 2 == 1, "The kernel size must be odd");

    template<typename T>
    elastic_distorter(const T& image) {
        static_assert(etl::dimensions<T>() == 3, "elastic_distorter can only be used with 3D images");

        cpp_unused(image);

        // Precompute the gaussian kernel

        auto gaussian = [](double x, double y) {
            auto Z = 2.0 * M_PI * sigma * sigma;
            return (1.0 / Z) * std::exp(-((x * x + y * y) / (2.0 * sigma * sigma)));
        };

        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                kernel(i, j) = gaussian(double(i) - mid, double(j) - mid);
            }
        }
    }

    size_t scaling() const {
        return 10;
    }

    template<typename O>
    void transform(O&& target){
        const size_t width  = etl::dim<1>(target);
        const size_t height = etl::dim<2>(target);

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

        // 2. Normalize and scale the displacement field

        d_x_blur *= (weight(8) / sum(d_x_blur));
        d_y_blur *= (weight(8) / sum(d_y_blur));

        // 3. Apply the displacement field (using bilinear interpolation)

        auto safe = [&](size_t channel, weight x, weight y) {
            if (x < 0 || y < 0 || x > width - 1 || y > height - 1) {
                return target(channel, 0UL, 0UL);
            } else {
                return target(channel, size_t(x), size_t(y));
            }
        };

        for (size_t channel = 0; channel < etl::dim<0>(target); ++channel) {
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

                    target(channel, x, y) = value;
                }
            }
        }
    }

    void gaussian_blur(const etl::dyn_matrix<weight>& d, etl::dyn_matrix<weight>& d_blur){
        const size_t width = etl::dim<0>(d);
        const size_t height = etl::dim<1>(d);

        for (size_t j = 0; j < width; ++j) {
            for (size_t k = 0; k < height; ++k) {
                weight sum(0.0);

                for (size_t p = 0; p < K; ++p) {
                    if (long(j) + p - mid >= 0 && long(j) + p - mid < width) {
                        for (size_t q = 0; q < K; ++q) {
                            if (long(k) + q - mid >= 0 && long(k) + q - mid < height) {
                                sum += kernel(p, q) * d(j + p - mid, k + q - mid);
                            }
                        }
                    }
                }

                d_blur(j, k) = d(j, k) - (sum / (K * K));
            }
        }
    }
};

template<typename Desc>
struct elastic_distorter <Desc, std::enable_if_t<!Desc::ElasticDistortion>> {
    template<typename T>
    elastic_distorter(const T& image){
        cpp_unused(image);
    }

    static constexpr size_t scaling() {
        return 1;
    }

    template<typename O>
    static void transform(O&& target){
        cpp_unused(target);
    }
};

} //end of dll namespace
