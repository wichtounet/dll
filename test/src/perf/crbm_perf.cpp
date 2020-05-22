//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "dll_test.hpp"

#include "dll/rbm/conv_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("crbm/mnist_140", "crbm::slow") {
    dll::conv_rbm_square_desc<
        2, 28, 40, 17,
        dll::batch_size<100>,
        dll::momentum, dll::weight_type<float>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 2, 28, 28>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);
    REQUIRE(error < 1);

    dll::dump_timers();
}

DLL_TEST_CASE("crbm/mnist_142", "crbm::slow_second") {
    dll::conv_rbm_square_desc<
        40, 12, 40, 7,
        dll::batch_size<100>,
        dll::momentum, dll::weight_type<float>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 40, 12, 12>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);
    REQUIRE(error < 1);
}

DLL_TEST_CASE("crbm/mnist_144", "crbm::slow") {
    dll::conv_rbm_square_desc<
        1, 28, 40, 5,
        dll::batch_size<100>,
        dll::momentum, dll::weight_type<float>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);
    REQUIRE(error < 1e-1);
}
