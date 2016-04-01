//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "dll/conv_rbm.hpp"
#include "dll/augment_layer.hpp"
#include "dll/patches_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/augment/mnist/1", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
            dll::augment_layer_desc<dll::copy<2>, dll::copy<3>>::layer_t,
            dll::conv_rbm_desc_square<1, 28, 20, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);
}

// This is here as a a test for multiplex layers
TEST_CASE("unit/augment/mnist/2", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::patches_layer_desc<14, 14, 14, 14>::layer_t
            , dll::conv_rbm_desc_square<1, 14, 10, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(20);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 2);
}

TEST_CASE("unit/augment/mnist/3", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::augment_layer_desc<dll::copy<2>, dll::copy<3>>::layer_t
            , dll::patches_layer_desc<14, 14, 14, 14>::layer_t
            , dll::conv_rbm_desc_square<1, 14, 10, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);
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

    //auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    //REQUIRE(!dataset.training_images.empty());

    //mnist::binarize_dataset(dataset);

    //auto dbn = std::make_unique<dbn_t>();

    //dbn->display();

    //dbn->pretrain(dataset.training_images, 20);
//}
