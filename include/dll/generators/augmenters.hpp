//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <atomic>
#include <thread>

#include "dll/util/random.hpp"

namespace dll {

/*!
 * \brief Randomly extract crops of a certain size from images
 */
template <typename Desc, typename Enable = void>
struct random_cropper;

/*!
 * \copydoc random_cropper
 */
template <typename Desc>
struct random_cropper<Desc, std::enable_if_t<Desc::random_crop_x != 0 && Desc::random_crop_y!= 0 >> {
    static constexpr size_t random_crop_x = Desc::random_crop_x; ///< The width of the crop
    static constexpr size_t random_crop_y = Desc::random_crop_y; ///< The height of the crop

    size_t x = 0; ///< The input image width
    size_t y = 0; ///< The input image height

    std::uniform_int_distribution<size_t> dist_x; ///< The distribution in x for picking the crop start
    std::uniform_int_distribution<size_t> dist_y; ///< The distribution in y for picking the crop start

    /*!
     * \brief Initialize the random_cropper
     * \param image The image to crop from
     */
    template <typename T>
    random_cropper(const T& image){
        static_assert(etl::dimensions<T>() == 3, "random_cropper can only be used with 3D images");

        y = etl::dim<1>(image);
        x = etl::dim<2>(image);

        dist_x = std::uniform_int_distribution<size_t>{0, x - random_crop_x};
        dist_y = std::uniform_int_distribution<size_t>{0, y - random_crop_y};
    }

    /*!
     * \brief The number of generated images from one input image
     * \return The augmentation factor
     */
    size_t scaling() const {
        return (x - random_crop_x) * (y - random_crop_y);
    }

    /*!
     * \brief Transform an image.
     *
     * This is used as the first step for data augmentation.
     *
     * \param target The target output
     * \param image The input image
     */
    template <typename O, typename T>
    void transform_first(O&& target, const T& image) {
        const size_t y_offset = dist_y(dll::rand_engine());
        const size_t x_offset = dist_x(dll::rand_engine());

        for (size_t c = 0; c < etl::dim<0>(image); ++c) {
            for (size_t y = 0; y < random_crop_y; ++y) {
                for (size_t x = 0; x < random_crop_x; ++x) {
                    target(c, y, x) = image(c, y_offset + y, x_offset + x);
                }
            }
        }
    }

    /*!
     * \brief Transform an image for test.
     *
     * This is used as the first step for data augmentation.
     *
     * \param target The target output
     * \param image The input image
     */
    template <typename O, typename T>
    void transform_first_test(O&& target, const T& image) {
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

/*!
 * \copydoc random_cropper
 */
template <typename Desc>
struct random_cropper<Desc, std::enable_if_t<Desc::random_crop_x == 0 || Desc::random_crop_y == 0>> {
    /*!
     * \brief Initialize the random_cropper
     * \param image The image to crop from
     */
    template <typename T>
    random_cropper([[maybe_unused]] const T& image) {}

    /*!
     * \brief The number of generated images from one input image
     * \return The augmentation factor
     */
    static constexpr size_t scaling() {
        return 1;
    }

    /*!
     * \brief Transform an image.
     *
     * This is used as the first step for data augmentation.
     *
     * \param target The target output
     * \param image The input image
     */
    template <typename O, typename T>
    void transform_first(O&& target, const T& image) {
        target = image;
    }

    /*!
     * \brief Transform an image for test.
     *
     * This is used as the first step for data augmentation.
     *
     * \param target The target output
     * \param image The input image
     */
    template <typename O, typename T>
    void transform_first_test(O&& target, const T& image) {
        target = image;
    }
};

/*!
 * \brief Image augmenter by random horizontal and/or vertical mirroring
 */
template <typename Desc, typename Enable = void>
struct random_mirrorer;

/*!
 * \copydoc random_mirrorer
 */
template <typename Desc>
struct random_mirrorer<Desc, std::enable_if_t<Desc::HorizontalMirroring || Desc::VerticalMirroring>> {
    static constexpr bool horizontal = Desc::HorizontalMirroring; ///< Indicates if random mirroring is done
    static constexpr bool vertical   = Desc::VerticalMirroring;   ///< Indicates if vertical mirroring is done

    std::uniform_int_distribution<size_t> dist; ///< The random distribution

    /*!
     * \brief Initialize the random_mirrorer
     * \param image The image to crop from
     */
    template <typename T>
    random_mirrorer([[maybe_unused]] const T& image){
        static_assert(etl::dimensions<T>() == 3, "random_mirrorer can only be used with 3D images");

        if (horizontal && vertical) {
            dist = std::uniform_int_distribution<size_t>{0, 3};
        } else {
            dist = std::uniform_int_distribution<size_t>{0, 2};
        }
    }

    /*!
     * \brief The number of generated images from one input image
     * \return The augmentation factor
     */
    size_t scaling() const {
        if (horizontal && vertical) {
            return 3;
        } else {
            return 2;
        }
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    void transform(O&& target) {
        auto choice = dist(dll::rand_engine());

        if (horizontal && vertical && choice == 1) {
            for (size_t c = 0; c < etl::dim<0>(target); ++c) {
                target(c) = vflip(target(c));
            }
        } else if (horizontal && vertical && choice == 2) {
            for (size_t c = 0; c < etl::dim<0>(target); ++c) {
                target(c) = hflip(target(c));
            }
        }

        if (horizontal && choice == 1) {
            for (size_t c = 0; c < etl::dim<0>(target); ++c) {
                target(c) = hflip(target(c));
            }
        }

        if (vertical && choice == 1) {
            for (size_t c = 0; c < etl::dim<0>(target); ++c) {
                target(c) = vflip(target(c));
            }
        }
    }
};

/*!
 * \copydoc random_mirrorer
 */
template <typename Desc>
struct random_mirrorer<Desc, std::enable_if_t<!Desc::HorizontalMirroring && !Desc::VerticalMirroring>> {
    /*!
     * \brief Initialize the random_mirrorer
     * \param image The image to crop from
     */
    template <typename T>
    random_mirrorer([[maybe_unused]] const T& image) {}

    /*!
     * \brief The number of generated images from one input image
     * \return The augmentation factor
     */
    static constexpr size_t scaling() {
        return 1;
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    static void transform([[maybe_unused]] O&& target) {}
};

/*!
 * \brief Data augmenter by noise
 */
template <typename Desc, typename Enable = void>
struct random_noise;

/*!
 * \copydoc random_noise
 */
template <typename Desc>
struct random_noise<Desc, std::enable_if_t<Desc::Noise != 0>> {
    static constexpr size_t N = Desc::Noise; ///< The amount of noise (in percent)

    std::uniform_int_distribution<size_t> dist; ///< The random distribution

    /*!
     * \brief Initialize the random_noise
     * \param image The image to crop from
     */
    template <typename T>
    random_noise([[maybe_unused]] const T& image) : dist(0, 1000) {}

    /*!
     * \brief The number of generated images from one input image
     * \return The augmentation factor
     */
    size_t scaling() const {
        return 10;
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    void transform(O&& target) {
        auto& g = dll::rand_engine();

        for (auto& v : target) {
            v *= dist(g) < N * 10 ? 0.0 : 1.0;
        }
    }
};

/*!
 * \copydoc random_noise
 */
template <typename Desc>
struct random_noise<Desc, std::enable_if_t<Desc::Noise == 0>> {
    /*!
     * \brief Initialize the random_noise
     * \param image The image to crop from
     */
    template <typename T>
    random_noise([[maybe_unused]] const T& image) {}

    /*!
     * \brief The number of generated images from one input image
     * \return The augmentation factor
     */
    static constexpr size_t scaling() {
        return 1;
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    static void transform([[maybe_unused]] O&& target) {}
};

/*!
 * \brief Elastic distorter for the images
 */
template <typename Desc, typename Enable = void>
struct elastic_distorter;

// TODO This needs to be made MUCH faster

/*!
 * \copydoc elastic_distorter
 */
template <typename Desc>
struct elastic_distorter<Desc, std::enable_if_t<Desc::ElasticDistortion != 0>> {
    using weight = typename Desc::weight; ///< The data type

    static constexpr size_t K     = Desc::ElasticDistortion;         ///< size of elastic distortion kernel
    static constexpr size_t mid   = K / 2;                           ///< Half of the kernel
    static constexpr double sigma = 0.8 + 0.3 * ((K - 1) * 0.5 - 1); ///< Sigma for gaussian kernel

    etl::fast_dyn_matrix<weight, K, K> kernel; ///< The precomputed kernel

    static_assert(K % 2 == 1, "The kernel size must be odd");

    /*!
     * \brief Initialize the elastic_distorter
     * \param image The image to distort
     */
    template <typename T>
    elastic_distorter([[maybe_unused]] const T& image) {
        static_assert(etl::dimensions<T>() == 3, "elastic_distorter can only be used with 3D images");

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

    /*!
     * \brief The number of generated images from one input image
     * \return The augmentation factor
     */
    size_t scaling() const {
        return 10;
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    void transform(O&& target) {
        const size_t width  = etl::dim<1>(target);
        const size_t height = etl::dim<2>(target);

        // 0. Generate random displacement fields

        etl::dyn_matrix<weight> d_x(width, height);
        etl::dyn_matrix<weight> d_y(width, height);

        d_x = etl::uniform_generator(dll::rand_engine(), -1.0, 1.0);
        d_y = etl::uniform_generator(dll::rand_engine(), -1.0, 1.0);

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

    /*!
     * \brief Apply a gaussian blur on the distortion matrix
     */
    void gaussian_blur(const etl::dyn_matrix<weight>& d, etl::dyn_matrix<weight>& d_blur) {
        const size_t width  = etl::dim<0>(d);
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

/*!
 * \copydoc elastic_distorter
 */
template <typename Desc>
struct elastic_distorter<Desc, std::enable_if_t<Desc::ElasticDistortion == 0>> {
    /*!
     * \brief Initialize the elastic_distorter
     * \param image The image to distort
     */
    template <typename T>
    elastic_distorter([[maybe_unused]] const T& image) {}

    /*!
     * \brief The number of generated images from one input image
     * \return The augmentation factor
     */
    static constexpr size_t scaling() {
        return 1;
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    static void transform([[maybe_unused]] O&& target) {}
};

} //end of dll namespace
