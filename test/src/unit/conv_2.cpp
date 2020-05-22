//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/conv/conv_layer.hpp"
#include "dll/neural/dense/dense_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/pooling/avgp_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("unit/conv/sgd/6", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::RELU>, dll::initializer<dll::init_he>>::layer_t,
            dll::conv_layer_desc<6, 24, 24, 6, 5, 5, dll::activation<dll::function::RELU>, dll::initializer<dll::init_he>>::layer_t,
            dll::dense_layer_desc<6 * 20 * 20, 200, dll::activation<dll::function::RELU>, dll::initializer<dll::init_he>>::layer_t,
            dll::dense_layer_desc<200, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::trainer<dll::sgd_trainer>, dll::batch_size<20>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(600);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate    = 0.001;
    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;

    FT_CHECK(50, 6e-2);
    TEST_CHECK(0.25);
}

DLL_TEST_CASE("unit/conv/sgd/7", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_2d_layer_desc<6, 24, 24, 2, 2>::layer_t,
            dll::conv_layer_desc<6, 12, 12, 5, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<5 * 10 * 10, 100, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::trainer<dll::sgd_trainer>, dll::batch_size<20>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(2000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.005;

    FT_CHECK(50, 6e-2);
    TEST_CHECK(0.25);
}

DLL_TEST_CASE("unit/conv/sgd/8", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer<1, 28, 28, 6, 5, 5, dll::relu>,
            dll::avgp_3d_layer<6, 24, 24, 1, 2, 2>,
            dll::conv_layer<6, 12, 12, 6, 3, 3, dll::relu>,
            dll::dense_layer<6 * 10 * 10, 100, dll::relu>,
            dll::dense_layer<100, 10, dll::softmax>
        >,
        dll::updater<dll::updater_type::ADAM>,
        dll::batch_size<20>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(2000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.001;

    FT_CHECK(25, 6e-2);
    TEST_CHECK(0.25);
}
