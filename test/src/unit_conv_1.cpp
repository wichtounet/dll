//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/dense_layer.hpp"
#include "dll/transform/scale_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/pooling/avgp_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/conv/sgd/1", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 6, 24, 24, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_desc<6 * 24 * 24, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.07;

    FT_CHECK(25, 5e-2);
    TEST_CHECK(0.2);
}

TEST_CASE("unit/conv/sgd/2", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 6, 24, 24, dll::activation<dll::function::TANH>>::layer_t,
            dll::dense_desc<6 * 24 * 24, 10, dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.10;

    FT_CHECK(25, 5e-2);
    TEST_CHECK(0.4);
}

TEST_CASE("unit/conv/sgd/3", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 6, 24, 24, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<6 * 24 * 24, 10, dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.07;

    FT_CHECK(25, 5e-2);
    TEST_CHECK(0.2);
}

TEST_CASE("unit/conv/sgd/4", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 6, 24, 24, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::conv_desc<6, 24, 24, 4, 20, 20, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_desc<4 * 20 * 20, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.1;

    FT_CHECK(25, 6e-2);
    TEST_CHECK(0.2);
}

TEST_CASE("unit/conv/sgd/5", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 24, 24, dll::activation<dll::function::RELU>>::layer_t,
            dll::conv_desc<10, 24, 24, 6, 20, 20, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<6 * 20 * 20, 200, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<200, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    FT_CHECK(25, 6e-2);
    TEST_CHECK(0.2);
}
