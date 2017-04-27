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
#include "dll/neural/activation_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/pooling/avgp_layer.hpp"

#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/conv/sgd/6", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::conv_desc<10, 24, 24, 6, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<6 * 20 * 20, 200, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<200, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::momentum, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate    = 0.008;
    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;

    FT_CHECK(25, 6e-2);
    TEST_CHECK(0.2);
}

TEST_CASE("unit/conv/sgd/7", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<6, 24, 24, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_desc<6, 12, 12, 5, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<5 * 8 * 8, 100, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.1;

    FT_CHECK(25, 6e-2);
    TEST_CHECK(0.2);
}

TEST_CASE("unit/conv/sgd/8", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 8, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::avgp_layer_3d_desc<8, 24, 24, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_desc<8, 12, 12, 6, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<6 * 8 * 8, 100, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.12;

    FT_CHECK(25, 6e-2);
    TEST_CHECK(0.25);
}
