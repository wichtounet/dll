//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/neural/conv/conv_layer.hpp"
#include "dll/neural/activation/activation_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/datasets.hpp"

// Fully-Connected Network on CIFAR-10

DLL_TEST_CASE("cifar/dense/sgd/1", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<3 * 32 * 32, 1000>::layer_t,
            dll::dense_layer_desc<1000, 500>::layer_t,
            dll::dense_layer_desc<500, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<20>>::dbn_t dbn_t;

    auto dataset = dll::make_cifar10_dataset_sub(2000, dll::batch_size<20>{});

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.01;
    dbn->momentum = 0.9;

    FT_CHECK_DATASET(50, 5e-2);
    TEST_CHECK_DATASET(0.2);
}

DLL_TEST_CASE("cifar/conv/sgd/1", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<3, 32, 32, 6, 5, 5>::layer_t,
            dll::conv_layer_desc<6, 28, 28, 6, 5, 5>::layer_t,
            dll::dense_layer_desc<6 * 24 * 24, 500>::layer_t,
            dll::dense_layer_desc<500, 10, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<20>>::dbn_t dbn_t;

    auto dataset = cifar::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 3, 32, 32>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.01;
    dbn->momentum = 0.9;

    FT_CHECK(50, 6e-2);
    TEST_CHECK(0.2);
}

DLL_TEST_CASE("cifar/conv/sgd/2", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<3, 32, 32, 12, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_3d_layer_desc<12, 28, 28, 1, 2, 2>::layer_t,
            dll::conv_layer_desc<12, 14, 14, 24, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_3d_layer_desc<24, 12, 12, 1, 2, 2>::layer_t,
            dll::dense_layer_desc<24 * 6 * 6, 64, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<64, 10, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<50>>::dbn_t dbn_t;

    auto dataset = cifar::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 3, 32, 32>>(5000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.001;
    dbn->momentum = 0.9;

    FT_CHECK(50, 6e-2);
    TEST_CHECK(0.2);
}
