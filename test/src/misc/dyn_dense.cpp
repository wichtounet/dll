//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/dense/dyn_dense_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dyn_dense/sgd/1", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_layer_desc<>::layer_t,
            dll::dyn_dense_layer_desc<>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("dyn_dense/sgd/2", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_layer_desc<dll::activation<dll::function::TANH>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("dyn_dense/sgd/3", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_layer_desc<>::layer_t,
            dll::dyn_dense_layer_desc<>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("dyn_dense/sgd/4", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.4);
}

DLL_TEST_CASE("dyn_dense/sgd/5", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_layer_desc<dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("dyn_dense/sgd/6", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_layer_desc<dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>, dll::scale_pre<255>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);

}

DLL_TEST_CASE("dyn_dense/sgd/7", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    dbn->display();

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("dyn_dense/sgd/8", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_dense_layer_desc<dll::activation<dll::function::TANH>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>, dll::scale_pre<255>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 10);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}
