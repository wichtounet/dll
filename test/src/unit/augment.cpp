//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/rbm/dyn_conv_rbm.hpp"
#include "dll/rbm/conv_rbm.hpp"
#include "dll/patches/patches_layer.hpp"
#include "dll/patches/dyn_patches_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// This is here as a a test for multiplex layers (compilation)
TEST_CASE("unit/augment/mnist/3", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::patches_layer_desc<14, 14, 14, 14>::layer_t
            , dll::conv_rbm_desc_square<1, 14, 10, 7, dll::momentum, dll::batch_size<10>>::layer_t
        >, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(20);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 2);

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() > 0);
}

TEST_CASE("unit/augment/mnist/10", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::dyn_patches_layer_desc<>::layer_t
            , dll::dyn_conv_rbm_desc<dll::momentum>::layer_t
        >, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template init_layer<0>(14, 14, 14, 14);
    dbn->template init_layer<1>(1, 14, 14, 10, 7, 7);

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() > 0);
}
