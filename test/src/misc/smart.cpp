//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/rbm/rbm.hpp"
#include "dll/rbm/conv_rbm.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

/*
 * These tests are ensuring compilation of functions gathering the features
 * on different types of network with several multiplex layers
 */

DLL_TEST_CASE("smart/mnist/1", "[dbn][smart]") {
    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t
        >, dll::batch_size<50>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    REQUIRE(dbn->forward_one(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->train_forward_one(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->test_forward_one(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->full_activation_probabilities(dataset.training_images[0]).size() == 100);
}

DLL_TEST_CASE("smart/mnist/2", "[dbn][smart]") {
    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t
        >, dll::batch_size<50>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    REQUIRE(dbn->forward_one(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->train_forward_one(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->test_forward_one(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->full_activation_probabilities(dataset.training_images[0]).size() == 300);
}
