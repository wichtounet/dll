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

TEST_CASE("rbm/mnist_3", "rbm::pcd_trainer") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::momentum,
        dll::trainer_rbm<dll::pcd1_trainer_t>>::rbm_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

//TODO Still not very convincing
TEST_CASE("rbm/mnist_15", "rbm::pcd_gaussian") {
    dll::rbm_desc<
        28 * 28, 144,
        dll::batch_size<25>,
        dll::momentum,
        dll::trainer_rbm<dll::pcd1_trainer_t>,
        dll::visible<dll::unit_type::GAUSSIAN>>::rbm_t rbm;

    rbm.learning_rate /= 20.0;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(500);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}
