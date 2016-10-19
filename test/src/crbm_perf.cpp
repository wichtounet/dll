//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "catch.hpp"

#include "dll/rbm/conv_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("crbm/mnist_140", "crbm::slow") {
    dll::conv_rbm_desc_square<
        2, 28, 40, 12,
        dll::batch_size<50>,
        dll::momentum, dll::weight_type<float>>::layer_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, float>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    for (auto& image : dataset.training_images) {
        image.reserve(image.size() * 2);
        auto end = image.size();
        for (std::size_t i = 0; i < end; ++i) {
            image.push_back(image[i]);
        }
    }

    auto error = rbm.train(dataset.training_images, 25);

    REQUIRE(error < 1e-1);
}

TEST_CASE("crbm/mnist_141", "crbm::slow_parallel") {
    dll::conv_rbm_desc_square<
        2, 28, 40, 12,
        dll::batch_size<50>,
        dll::momentum,
        dll::parallel_mode, dll::weight_type<float>>::layer_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, float>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    for (auto& image : dataset.training_images) {
        image.reserve(image.size() * 2);
        auto end = image.size();
        for (std::size_t i = 0; i < end; ++i) {
            image.push_back(image[i]);
        }
    }

    auto error = rbm.train(dataset.training_images, 25);

    REQUIRE(error < 1e-1);
}

TEST_CASE("crbm/mnist_142", "crbm::slow_second") {
    dll::conv_rbm_desc_square<
        40, 12, 40, 6,
        dll::batch_size<25>,
        dll::momentum, dll::weight_type<float>>::layer_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, float>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    for (auto& image : dataset.training_images) {
        image.reserve(image.size() * 40);
        auto end = image.size();
        for (std::size_t i = 0; i < end; ++i) {
            image.push_back(image[i]);
        }
        image.resize(12 * 12 * 40);
    }

    auto error = rbm.train(dataset.training_images, 25);

    REQUIRE(error < 1e-1);
}

TEST_CASE("crbm/mnist_143", "crbm::slow_parallel_second") {
    dll::conv_rbm_desc_square<
        40, 12, 40, 6,
        dll::batch_size<25>,
        dll::momentum,
        dll::parallel_mode, dll::weight_type<float>>::layer_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, float>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    for (auto& image : dataset.training_images) {
        image.reserve(image.size() * 40);
        auto end = image.size();
        for (std::size_t i = 0; i < end; ++i) {
            image.push_back(image[i]);
        }
        image.resize(12 * 12 * 40);
    }

    auto error = rbm.train(dataset.training_images, 25);

    REQUIRE(error < 1e-1);
}

TEST_CASE("crbm/mnist_144", "crbm::slow") {
    dll::conv_rbm_desc_square<
        1, 28, 40, 24,
        dll::batch_size<25>,
        dll::momentum, dll::weight_type<float>>::layer_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, float>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);

    REQUIRE(error < 1e-1);
}

TEST_CASE("crbm/mnist_145", "crbm::slow") {
    dll::conv_rbm_desc_square<
        1, 28, 40, 24,
        dll::batch_size<25>,
        dll::momentum, dll::parallel_mode, dll::weight_type<float>>::layer_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, float>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);

    REQUIRE(error < 1e-1);
}
