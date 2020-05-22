//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/rbm/dyn_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

//Only here for benchmarking purposes
DLL_TEST_CASE("dyn_rbm/mnist_14", "rbm::slow") {
    dll::dyn_rbm_desc<>::layer_t rbm(28 * 28, 400);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 10);

    REQUIRE(error < 5e-2);
}
