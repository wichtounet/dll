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

#include "dll/transform/scale_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/conv/sgd/9", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::scale_layer_desc<1, 256>::layer_t,
            dll::conv_desc<1, 28, 28, 5, 5, 5, dll::activation<dll::function::TANH>>::layer_t,
            dll::dense_desc<5 * 24 * 24, 10, dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    FT_CHECK(25, 6e-2);
    TEST_CHECK(0.3);
}

TEST_CASE("unit/conv/sgd/10", "[unit][conv][dbn][mnist][sgd]") {
    //Note: This is a reduced lenet version
    //Note: The activation layers are here to ensure compilation
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 5, 5, dll::activation<dll::function::IDENTITY>>::layer_t,
            dll::activation_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<10, 24, 24, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_desc<10, 12, 12, 25, 5, 5, dll::activation<dll::function::IDENTITY>>::layer_t,
            dll::activation_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<25, 8, 8, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::dense_desc<25 * 4 * 4, 500, dll::activation<dll::function::IDENTITY>>::layer_t,
            dll::activation_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<500, 10, dll::activation<dll::function::IDENTITY>>::layer_t,
            dll::activation_layer_desc<dll::activation<dll::function::SOFTMAX>>::layer_t
        >, dll::momentum, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->l2_weight_cost   = 0.0005;
    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK(25, 6e-2);
    TEST_CHECK(0.2);
}
