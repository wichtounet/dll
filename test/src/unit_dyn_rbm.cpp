//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "dll/dyn_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/dyn_rbm/mnist/1", "[rbm][dyn][momentum][unit]") {
    dll::dyn_rbm_desc<
        dll::momentum>::layer_t rbm(28 * 28, 100);

    rbm.batch_size = 25;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 5e-2);
}

TEST_CASE("unit/dyn_rbm/mnist/2", "[rbm][dyn][gaussian][momentum][unit]") {
    dll::dyn_rbm_desc<
        dll::visible<dll::unit_type::GAUSSIAN>,
        dll::momentum>::layer_t rbm(28 * 28, 100);

    rbm.learning_rate *= 10;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(75);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 1e-1);
}

TEST_CASE("unit/dyn_rbm/mnist/3", "[rbm][dyn][relu][momentum][unit]") {
    dll::dyn_rbm_desc<
        dll::hidden<dll::unit_type::RELU>,
        dll::momentum>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 5e-2);
}
