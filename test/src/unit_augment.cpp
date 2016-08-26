//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "dll/dyn_conv_rbm.hpp"
#include "dll/conv_rbm.hpp"
#include "dll/augment_layer.hpp"
#include "dll/patches_layer.hpp"
#include "dll/dyn_patches_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/augment/mnist/1", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
            dll::augment_layer_desc<dll::copy<2>, dll::copy<3>>::layer_t,
            dll::conv_rbm_desc_square<1, 28, 20, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() > 0);
}

TEST_CASE("unit/augment/mnist/2", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
            dll::augment_layer_desc<dll::copy<2>, dll::copy<3>>::layer_t,
            dll::conv_rbm_desc_square<1, 28, 20, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >, dll::batch_mode>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() > 0);
}

// This is here as a a test for multiplex layers (compilation)
TEST_CASE("unit/augment/mnist/3", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::patches_layer_desc<14, 14, 14, 14>::layer_t
            , dll::conv_rbm_desc_square<1, 14, 10, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(20);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 2);

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() > 0);
}

TEST_CASE("unit/augment/mnist/4", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::augment_layer_desc<dll::copy<2>, dll::copy<3>>::layer_t
            , dll::patches_layer_desc<14, 14, 14, 14>::layer_t
            , dll::conv_rbm_desc_square<1, 14, 10, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    //TODO REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() > 0);
    // This does not work since it will distort the images :(
}

TEST_CASE("unit/augment/mnist/5", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::augment_layer_desc<dll::copy<2>, dll::copy<3>>::layer_t
            , dll::patches_layer_desc<14, 14, 14, 14>::layer_t
            , dll::conv_rbm_desc_square<1, 14, 10, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >, dll::batch_mode>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);
}

TEST_CASE("unit/augment/mnist/6", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::augment_layer_desc<dll::elastic<3>>::layer_t
            , dll::patches_layer_desc<14, 14, 14, 14>::layer_t
            , dll::conv_rbm_desc_square<1, 14, 10, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);
}

TEST_CASE("unit/augment/mnist/7", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::augment_layer_desc<dll::elastic<3>>::layer_t
            , dll::patches_layer_desc<14, 14, 14, 14>::layer_t
            , dll::conv_rbm_desc_square<1, 14, 10, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >, dll::batch_mode>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);
}

TEST_CASE("unit/augment/mnist/8", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::augment_layer_desc<dll::elastic<3>>::layer_t
            , dll::patches_layer_desc<14, 14, 14, 14>::layer_t
            , dll::conv_rbm_desc_square<1, 14, 10, 8, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::conv_rbm_desc_square<10, 8, 10, 6, dll::momentum, dll::batch_size<10>>::layer_t
        >, dll::batch_mode>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);
}

TEST_CASE("unit/augment/mnist/9", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::augment_layer_desc<dll::elastic<3>>::layer_t
            , dll::dyn_conv_rbm_desc<dll::momentum>::layer_t
            , dll::dyn_conv_rbm_desc<dll::momentum>::layer_t
        >, dll::batch_mode>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template init_layer<1>(1, 28, 28, 10, 20, 20);
    dbn->template init_layer<2>(10, 20, 20, 10, 16, 16);

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);
}

TEST_CASE("unit/augment/mnist/10", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::dyn_patches_layer_desc<>::layer_t
            , dll::dyn_conv_rbm_desc<dll::momentum>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template init_layer<0>(14, 14, 14, 14);
    dbn->template init_layer<1>(1, 14, 14, 10, 8, 8);

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() > 0);
}

//TODO Make this work (harder than it seems with current architecture)
//TEST_CASE("unit/augment/mnist/100", "[cdbn][augment][unit]") {
    //using dbn_t =
        //dll::dbn_desc<dll::dbn_layers<
              //dll::augment_layer_desc<dll::copy<2>, dll::copy<1>>::layer_t
            //, dll::conv_rbm_desc_square<1, 28, 20, 20, dll::momentum, dll::batch_size<10>>::layer_t
            //, dll::augment_layer_desc<dll::copy<2>>::layer_t
            //, dll::conv_rbm_desc_square<20, 20, 10, 14, dll::momentum, dll::batch_size<10>>::layer_t
        //>>::dbn_t;

    //auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    //REQUIRE(!dataset.training_images.empty());

    //mnist::binarize_dataset(dataset);

    //auto dbn = std::make_unique<dbn_t>();

    //dbn->display();

    //dbn->pretrain(dataset.training_images, 20);
//}
