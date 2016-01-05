//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "catch.hpp"

#include "cpp_utils/data.hpp"

#include "dll/conv_rbm_mp.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/crbm_mp/mnist/1", "[crbm_mp][unit]") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 20, 12, 2,
        dll::batch_size<25>,
        dll::momentum,
        dll::serial,
        dll::parallel_mode>::layer_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 40);
    REQUIRE(error < 5e-2);

    rbm.v1 = dataset.training_images[1];

    rbm.template activate_hidden<true, false>(rbm.h1_a, rbm.h1_a, rbm.v1, rbm.v1);

    auto energy = rbm.energy(dataset.training_images[1], rbm.h1_a);
    REQUIRE(energy < 0.0);

    auto free_energy = rbm.free_energy();
    REQUIRE(free_energy < 0.0);
}

TEST_CASE("unit/crbm_mp/mnist/2", "[crbm_mp][gaussian][unit]") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 24, 2,
        dll::batch_size<25>,
        dll::momentum,
        dll::weight_decay<>,
        dll::visible<dll::unit_type::GAUSSIAN>>::layer_t rbm;

    rbm.learning_rate *= 2.0;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 1e-1);
}

TEST_CASE("unit/crbm_mp/mnist/3", "[crbm_mp][multic][unit]") {
    dll::conv_rbm_mp_desc_square<
        2, 28, 20, 12, 2,
        dll::batch_size<25>,
        dll::momentum>::layer_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    for (auto& image : dataset.training_images) {
        image.reserve(image.size() * 2);
        auto end = image.size();
        for (std::size_t i = 0; i < end; ++i) {
            image.push_back(image[i]);
        }
    }

    auto error = rbm.train(dataset.training_images, 50);

    REQUIRE(error < 5e-2);
}

TEST_CASE("unit/crbm_mp/mnist/4", "[crbm_mp][denoising][unit]") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 30, 12, 2,
        dll::batch_size<25>,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::visible<dll::unit_type::GAUSSIAN>,
        dll::shuffle>::layer_t rbm;

    rbm.learning_rate *= 4;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto noisy = dataset.training_images;

    std::default_random_engine rand_engine(56);
    std::normal_distribution<double> normal_distribution(0.0, 0.1);
    auto noise = std::bind(normal_distribution, rand_engine);

    for (auto& image : noisy) {
        for (auto& noisy_x : image) {
            noisy_x += noise();
        }
    }

    cpp::normalize_each(noisy);

    auto error = rbm.train_denoising(noisy, dataset.training_images, 50);
    REQUIRE(error < 1e-1);
    cpp_unused(error);
}

TEST_CASE("unit/crbm_mp/mnist/5", "[crbm_mp][relu][unit]") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 20, 12, 2,
        dll::batch_size<5>,
        dll::hidden<dll::unit_type::RELU>>::layer_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 5e-2);
}

TEST_CASE("unit/crbm_mp/mnist/6", "[crbm_mp][lee][unit]") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 20, 12, 2,
        dll::batch_size<10>,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::sparsity<dll::sparsity_method::LEE>,
        dll::bias<dll::bias_mode::SIMPLE>>::layer_t rbm;

    rbm.l2_weight_cost = 0.01;
    rbm.learning_rate  = 0.01;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);
    REQUIRE(error < 3e-2);
}

TEST_CASE("unit/crbm_mp/mnist/7", "[crbm_mp][lee][gaussian][unit]") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 20, 2,
        dll::batch_size<10>,
        dll::momentum,
        dll::visible<dll::unit_type::GAUSSIAN>,
        dll::weight_decay<dll::decay_type::L2>,
        dll::sparsity<dll::sparsity_method::LEE>,
        dll::bias<dll::bias_mode::SIMPLE>>::layer_t rbm;

    rbm.pbias        = 0.01;
    rbm.pbias_lambda = 0.1;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 9e-2);
}
