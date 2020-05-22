//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dense/ae/1", "[dense][dbn][mnist][sgd][ae]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 200>::layer_t,
            dll::dense_layer_desc<200, 28 * 28>::layer_t
        >, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.1;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set_ae(*dbn, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

DLL_TEST_CASE("dense/ae/2", "[dense][dbn][mnist][sgd][ae]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 200, dll::activation<dll::function::TANH>>::layer_t,
            dll::dense_layer_desc<200, 28 * 28, dll::activation<dll::function::TANH>>::layer_t
        >, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.1;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set_ae(*dbn, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

DLL_TEST_CASE("dense/ae/3", "[dense][dbn][mnist][sgd][ae]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 200, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<200, 28 * 28, dll::activation<dll::function::RELU>>::layer_t
        >, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.1;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set_ae(*dbn, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

DLL_TEST_CASE("dense/ae/4", "[dense][dbn][mnist][sgd][ae]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 200>::layer_t,
            dll::dense_layer_desc<200, 300>::layer_t,
            dll::dense_layer_desc<300, 28 * 28>::layer_t
        >, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.1;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set_ae(*dbn, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

DLL_TEST_CASE("dense/ae/5", "[dense][dbn][mnist][sgd][ae][momentum]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 200>::layer_t,
            dll::dense_layer_desc<200, 300>::layer_t,
            dll::dense_layer_desc<300, 28 * 28>::layer_t
        >, dll::updater<dll::updater_type::MOMENTUM>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.1;
    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set_ae(*dbn, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

DLL_TEST_CASE("dense/ae/6", "[dense][dbn][mnist][sgd][ae][momentum][decay]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 200>::layer_t,
            dll::dense_layer_desc<200, 300>::layer_t,
            dll::dense_layer_desc<300, 28 * 28>::layer_t
        >, dll::updater<dll::updater_type::MOMENTUM>, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.1;
    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set_ae(*dbn, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}
