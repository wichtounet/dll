//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "dll/rbm/rbm.hpp"
#include "dll/rbm/conv_rbm.hpp"
#include "dll/augment/augment_layer.hpp"
#include "dll/transform/scale_layer.hpp"
#include "dll/patches/patches_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

/*
 * These tests are ensuring compilation of functions gathering the features
 * on different types of network with several multiplex layers
 */

TEST_CASE("smart/mnist/1", "[dbn][smart]") {
    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t
        >, dll::batch_size<50>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->train_activation_probabilities(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->test_activation_probabilities(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->full_activation_probabilities(dataset.training_images[0]).size() == 100);
}

TEST_CASE("smart/mnist/2", "[dbn][smart]") {
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

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->train_activation_probabilities(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->test_activation_probabilities(dataset.training_images[0]).size() == 100);
    REQUIRE(dbn->full_activation_probabilities(dataset.training_images[0]).size() == 300);
}

TEST_CASE("smart/mnist/3", "[smart][cdbn][augment]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
            dll::augment_layer_desc<dll::copy<2>, dll::copy<3>>::layer_t,
            dll::conv_rbm_desc_square<1, 28, 8, 9, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() == 6);
    REQUIRE(dbn->train_activation_probabilities(dataset.training_images[0]).size() == 6);
    REQUIRE(dbn->test_activation_probabilities(dataset.training_images[0]).size() == 8 * 20 * 20);
}

TEST_CASE("smart/mnist/4", "[smart][cdbn][augment]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
            dll::scale_layer_desc<1, 256>::layer_t,
            dll::augment_layer_desc<dll::copy<2>, dll::copy<3>>::layer_t,
            dll::scale_layer_desc<1, 256>::layer_t,
            dll::conv_rbm_desc_square<1, 28, 8, 9, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() == 6);
    REQUIRE(dbn->train_activation_probabilities(dataset.training_images[0]).size() == 6);
    REQUIRE(dbn->test_activation_probabilities(dataset.training_images[0]).size() == 8 * 20 * 20);
}

TEST_CASE("smart/mnist/5", "[smart][cdbn][augment]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
            dll::augment_layer_desc<dll::copy<2>, dll::copy<3>>::layer_t,
            dll::patches_layer_desc<14, 14, 14, 14>::layer_t,
            dll::augment_layer_desc<dll::elastic<4>>::layer_t,
            dll::conv_rbm_desc_square<1, 14, 10, 7, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::conv_rbm_desc_square<10, 8, 10, 3, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() == 120);
    REQUIRE(dbn->train_activation_probabilities(dataset.training_images[0]).size() == 120);
    REQUIRE(dbn->test_activation_probabilities(dataset.training_images[0]).size() == 4);
}
