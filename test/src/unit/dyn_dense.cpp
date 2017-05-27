//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/dyn_dense_layer.hpp"
#include "dll/transform/shape_layer_1d.hpp"
#include "dll/transform/scale_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// Test Sigmoid -> Sigmoid network
TEST_CASE("unit/dyn_dense/sgd/1", "[unit][dyn_dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_desc<>::layer_t,
            dll::dyn_dense_desc<>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>, dll::normalize_pre>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 150);
    dbn->template layer_get<1>().init_layer(150, 10);

    dbn->learning_rate = 0.03;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.3);
}

// Test tanh -> tanh network
TEST_CASE("unit/dyn_dense/sgd/2", "[unit][dyn_dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_desc<dll::activation<dll::function::TANH>>::layer_t,
            dll::dyn_dense_desc<dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    dbn->learning_rate = 0.05;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.3);
}

// test momentum and weight decay
TEST_CASE("unit/dyn_dense/sgd/3", "[unit][dyn_dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_desc<>::layer_t,
            dll::dyn_dense_desc<>::layer_t>,
        dll::momentum, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 125);
    dbn->template layer_get<1>().init_layer(125, 10);

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.03;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// Test Sigmoid -> Softmax network
TEST_CASE("unit/dyn_dense/sgd/4", "[unit][dyn_dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_desc<dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dyn_dense_desc<dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::momentum, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// Test scale layer
TEST_CASE("unit/dyn_dense/sgd/5", "[unit][dyn_dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::shape_layer_1d_desc<28 * 28>::layer_t,
            dll::scale_layer_desc<1, 256>::layer_t,
            dll::dyn_dense_desc<dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dyn_dense_desc<dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::momentum, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<2>().init_layer(28 * 28, 100);
    dbn->template layer_get<3>().init_layer(100, 10);

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// Test Relu -> Softmax network
TEST_CASE("unit/dyn_dense/sgd/6", "[unit][dyn_dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_desc<dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::momentum, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// Test Relu -> Relu -> Softmax network
TEST_CASE("unit/dyn_dense/sgd/7", "[unit][dyn_dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_desc<dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::momentum, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 150);
    dbn->template layer_get<1>().init_layer(150, 150);
    dbn->template layer_get<2>().init_layer(150, 10);

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}
