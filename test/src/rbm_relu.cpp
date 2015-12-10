//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "catch.hpp"

#include "dll/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("rbm/mnist_9", "rbm::relu") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-2);
}

TEST_CASE("rbm/mnist_10", "rbm::relu1") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU1>>::layer_t rbm;

    rbm.learning_rate *= 2.0;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

TEST_CASE("rbm/mnist_11", "rbm::relu6") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU6>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}
