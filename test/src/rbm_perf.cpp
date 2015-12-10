//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "dll/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

//Only here for debugging purposes
TEST_CASE("rbm/mnist_18", "rbm::fast") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<5>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(25);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 5);

    REQUIRE(error < 5e-1);
}

TEST_CASE("rbm/mnist_101", "rbm::slow") {
    dll::rbm_desc<
        28 * 28, 459,
        dll::batch_size<48>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(1099);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 15);

    REQUIRE(error < 5e-2);
}

TEST_CASE("rbm/mnist_102", "rbm::slow_parallel") {
    dll::rbm_desc<
        28 * 28, 459,
        dll::batch_size<48>,
        dll::parallel_mode>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(1099);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 15);

    REQUIRE(error < 5e-2);
}
