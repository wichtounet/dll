//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/dense_layer.hpp"
#include "dll/scale_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// Test Sigmoid -> Sigmoid network
TEST_CASE( "unit/dense/sgd/1", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 150>::layer_t,
            dll::dense_desc<150, 10>::layer_t
        >,
        dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.03;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.3);
}

// Test tanh -> tanh network
TEST_CASE( "unit/dense/sgd/2", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100, dll::activation<dll::function::TANH>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::TANH>>::layer_t
        >,
        dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.3);
}

// Test momentum
TEST_CASE( "unit/dense/sgd/3", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100>::layer_t,
            dll::dense_desc<100, 10>::layer_t
        >
        , dll::momentum
        , dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.03;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// test momentum and weight decay
TEST_CASE( "unit/dense/sgd/4", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 125>::layer_t,
            dll::dense_desc<125, 10>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.03;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// Test tanh with momentum and weight decay
TEST_CASE( "unit/dense/sgd/5", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 175, dll::activation<dll::function::TANH>>::layer_t,
            dll::dense_desc<175, 10, dll::activation<dll::function::TANH>>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.005;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.3);
}

// Test identity activation function
TEST_CASE( "unit/dense/sgd/6", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100, dll::activation<dll::function::IDENTITY>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::IDENTITY>>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.4);
}

// Test ReLU -> Sigmoid network
TEST_CASE( "unit/dense/sgd/7", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::SIGMOID>>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// Test Sigmoid -> Softmax network
TEST_CASE( "unit/dense/sgd/8", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::SOFTMAX>>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// Test scale layer
TEST_CASE( "unit/dense/sgd/9", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::scale_layer_desc<1, 256>::layer_t,
            dll::dense_desc<28 * 28, 100, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::SOFTMAX>>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// Test Relu -> Softmax network
TEST_CASE( "unit/dense/sgd/10", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::SOFTMAX>>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// Test Relu -> Relu -> Softmax network
TEST_CASE( "unit/dense/sgd/11", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 150, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<150, 150, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<150, 10, dll::activation<dll::function::SOFTMAX>>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

// Test Sigmoid -> Sigmoid network with bold driver
TEST_CASE( "unit/dense/sgd/12", "[unit][dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 150>::layer_t,
            dll::dense_desc<150, 10>::layer_t
        >,
          dll::trainer<dll::sgd_trainer>
        , dll::lr_driver<dll::lr_driver_type::BOLD>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.1;

    FT_CHECK(100, 5e-2);
    TEST_CHECK(0.3);
}
