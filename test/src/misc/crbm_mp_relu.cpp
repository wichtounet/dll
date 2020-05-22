//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "dll_test.hpp"

#include "dll/rbm/conv_rbm_mp.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("crbm_mp/mnist_7", "crbm::relu") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 17, 2,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU>>::layer_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

DLL_TEST_CASE("crbm_mp/mnist_8", "crbm::relu1") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 17, 2,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU1>>::layer_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

DLL_TEST_CASE("crbm_mp/mnist_9", "crbm::relu6") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 17, 2,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU6>>::layer_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}
