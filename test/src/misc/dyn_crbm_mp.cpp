//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "dll_test.hpp"

#include "dll/rbm/dyn_conv_rbm_mp.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dyn_crbm_mp/mnist_1", "crbm::simple") {
    dll::dyn_conv_rbm_mp_desc<>::layer_t rbm;

    rbm.init_layer(1, 28, 28, 40, 12, 12, 2);
    rbm.learning_rate = 0.01;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);
    REQUIRE(error < 1e-1);
}

DLL_TEST_CASE("dyn_crbm_mp/mnist_2", "crbm::momentum") {
    dll::dyn_conv_rbm_mp_desc<dll::momentum>::layer_t rbm;

    rbm.init_layer(1, 28, 28, 40, 12, 12, 2);
    rbm.learning_rate = 0.01;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);
    REQUIRE(error < 1e-2);
}
