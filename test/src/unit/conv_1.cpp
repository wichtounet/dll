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
#include "dll/datasets.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("unit/conv/sgd/1", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_layer_desc<6 * 24 * 24, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<10>>::dbn_t dbn_t;

    // Load the dataset
    auto dataset = dll::make_mnist_dataset_sub(0, 500, dll::batch_size<10>{});

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    FT_CHECK_DATASET(25, 5e-2);
    TEST_CHECK_DATASET(0.25);
}

DLL_TEST_CASE("unit/conv/sgd/2", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::TANH>>::layer_t,
            dll::dense_layer_desc<6 * 24 * 24, 10, dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    FT_CHECK(25, 5e-2);
    TEST_CHECK(0.4);
}

DLL_TEST_CASE("unit/conv/sgd/3", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 4, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<4 * 24 * 24, 10, dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::SGD>, dll::batch_size<20>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(800);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    FT_CHECK(75, 6e-2);
    TEST_CHECK(0.25);
}

DLL_TEST_CASE("unit/conv/sgd/4", "[unit][conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::conv_layer_desc<6, 24, 24, 4, 5, 5, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_layer_desc<4 * 20 * 20, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<20>, dll::scale_pre<255>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(800);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.1;

    FT_CHECK(35, 0.2);
    TEST_CHECK(0.25);
}

DLL_TEST_CASE("unit/conv/sgd/5", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::RELU>, dll::initializer<dll::init_he>>::layer_t,
            dll::conv_layer_desc<6, 24, 24, 4, 5, 5, dll::activation<dll::function::RELU>, dll::initializer<dll::init_he>>::layer_t,
            dll::dense_layer_desc<4 * 20 * 20, 200, dll::activation<dll::function::RELU>, dll::initializer<dll::init_he>>::layer_t,
            dll::dense_layer_desc<200, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<20>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    FT_CHECK(25, 6e-2);
    TEST_CHECK(0.2);
}

// Test custom training
DLL_TEST_CASE("unit/conv/sgd/partial/1", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 6, 5, 5, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_layer_desc<6 * 24 * 24, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    using generator_t = dll::inmemory_data_generator_desc<dll::batch_size<10>, dll::categorical>;

    auto generator = dll::make_generator(dataset.training_images, dataset.training_labels, 10, generator_t{});

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.07;

    auto trainer = dbn->get_trainer();

    trainer.start_training(*dbn, 30);

    // Train for 25 epochs
    for(size_t epoch = 0; epoch < 30; ++epoch){
        trainer.start_epoch(*dbn, epoch);

        generator->reset();

        // Train for one epoch
        auto [error, loss] = trainer.train_epoch(*dbn, *generator, epoch);

        if(trainer.stop_epoch(*dbn, epoch, error, loss)){
            break;
        }
    }

    auto ft_error = trainer.stop_training(*dbn, 30, 30);

    REQUIRE(ft_error < 5e-2);

    TEST_CHECK(0.25);
}
