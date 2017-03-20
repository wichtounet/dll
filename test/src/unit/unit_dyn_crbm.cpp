//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "catch.hpp"

#include "cpp_utils/data.hpp"

#include "dll/rbm/dyn_conv_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/dyn_crbm/mnist/1", "[dyn_crbm][unit]") {
    dll::dyn_conv_rbm_desc<
        dll::weight_decay<dll::decay_type::L2_FULL>,
        dll::momentum>::layer_t rbm;

    rbm.init_layer(1, 28, 28, 20, 17, 17);

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);
    REQUIRE(error < 5e-2);

    rbm.v1 = dataset.training_images[1];

    rbm.template activate_hidden<true, false>(rbm.h1_a, rbm.h1_a, rbm.v1, rbm.v1);

    auto energy = rbm.energy(dataset.training_images[1], rbm.h1_a);
    REQUIRE(energy < 0.0);

    auto free_energy = rbm.free_energy();
    REQUIRE(free_energy < 0.0);
}

TEST_CASE("unit/dyn_crbm/mnist/2", "[dyn_crbm][parallel][unit]") {
    dll::dyn_conv_rbm_desc<
        dll::momentum,
        dll::parallel_mode,
        dll::weight_decay<dll::decay_type::L2>,
        dll::visible<dll::unit_type::GAUSSIAN>>::layer_t rbm;

    rbm.init_layer(1, 28, 28, 5, 5, 5);

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);
    REQUIRE(error < 0.1);
}

TEST_CASE("unit/dyn_crbm/mnist/3", "[dyn_crbm][unit]") {
    dll::dyn_conv_rbm_desc<dll::momentum>::layer_t rbm;

    rbm.init_layer(2, 28, 28, 20, 17, 17);

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    for (auto& image : dataset.training_images) {
        image.reserve(image.size() * 2);
        auto end = image.size();
        for (std::size_t i = 0; i < end; ++i) {
            image.push_back(image[i]);
        }
    }

    auto error = rbm.train(dataset.training_images, 20);

    REQUIRE(error < 5e-2);
}

TEST_CASE("unit/dyn_crbm/mnist/4", "[dyn_crbm][unit]") {
    dll::dyn_conv_rbm_desc<
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::shuffle,
        dll::hidden<dll::unit_type::RELU>>::layer_t rbm;

    rbm.learning_rate *= 5;
    rbm.init_layer(1, 28, 28, 40, 9, 9);

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);
    REQUIRE(error < 5e-2);
}
