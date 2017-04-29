//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "catch.hpp"

#include "dll/rbm/dyn_conv_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("dyn_crbm/mnist_1", "dyn_crbm::simple") {
    dll::dyn_conv_rbm_desc<>::layer_t rbm;

    rbm.init_layer(1, 28, 28, 40, 17, 17);
    rbm.batch_size = 25;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 2e-2);
}

TEST_CASE("dyn_crbm/mnist_2", "crbm::momentum") {
    dll::dyn_conv_rbm_desc<dll::momentum>::layer_t rbm;

    rbm.init_layer(1, 28, 28, 40, 17, 17);
    rbm.batch_size = 25;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}
