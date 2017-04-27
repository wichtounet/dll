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
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/conv/sgd/1", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::SIGMOID>>::layer_t,
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
            dll::conv_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::TANH>>::layer_t,
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
            dll::conv_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
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
            dll::conv_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::conv_desc<6, 24, 24, 4, 5, 5, dll::activation<dll::function::SIGMOID>>::layer_t,
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
            dll::conv_desc<1, 28, 28, 8, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::conv_desc<8, 24, 24, 6, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
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

// Test custom training
TEST_CASE("unit/conv/sgd/partial/1", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_desc<6 * 24 * 24, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.07;

    auto trainer = dbn->get_trainer();

    trainer.start_training(*dbn, false, 25);

    // Train for 25 epochs
    for(size_t epoch = 0; epoch < 25; ++epoch){
        trainer.start_epoch(*dbn, epoch);

        double error = 0.0;
        double loss = 0.0;

        for(std::size_t i = 0; i < 7; ++i){
            std::tie(loss, error) = trainer.train_partial(
                *dbn, false,
                dataset.training_images.begin() + i * 50, dataset.training_images.begin() + (i + 1) * 50,
                dataset.training_labels.begin() + i * 50,
                epoch);
        }

        if(trainer.stop_epoch(*dbn, epoch, error, loss)){
            break;
        }
    }

    auto ft_error = trainer.stop_training(*dbn);

    REQUIRE(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}
