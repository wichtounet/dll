//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/rbm/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("rbm/mnist_9", "rbm::relu") {
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

DLL_TEST_CASE("rbm/mnist_10", "rbm::relu1") {
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

DLL_TEST_CASE("rbm/mnist_11", "rbm::relu6") {
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

DLL_TEST_CASE("rbm/relu/1", "[relu][rbm][clip]") {
    using rbm_t = dll::rbm_desc<
            28 * 28, 100
            , dll::momentum
            , dll::batch_size<25>
            , dll::hidden<dll::unit_type::RELU>
            , dll::clip_gradients
        >::layer_t;

    rbm_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(500);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    rbm.initial_momentum = 0.9;
    rbm.momentum = 0.9;

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}
