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

TEST_CASE("dbn/ae/1", "[rbm][dbn][mnist][sgd][ae]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 28 * 28, dll::momentum, dll::batch_size<25>>::layer_t
        >, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 50);

    dbn->learning_rate = 0.1;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set_ae(*dbn, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

//TODO: This test does not work (pretraining seems to break havoc)
TEST_CASE("dbn/ae/2", "[rbm][dbn][mnist][sgd][ae]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200, dll::hidden<dll::unit_type::RELU>>::layer_t,
            dll::rbm_desc<200, 28 * 28, dll::hidden<dll::unit_type::RELU>>::layer_t
        >, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 50);

    dbn->learning_rate = 0.1;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set_ae(*dbn, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE("dbn/ae/3", "[rbm][dbn][mnist][sgd][ae]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200>::layer_t,
            dll::rbm_desc<200, 300>::layer_t,
            dll::rbm_desc<300, 28 * 28>::layer_t
        >, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 50);

    dbn->learning_rate = 0.1;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set_ae(*dbn, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE("dbn/ae/4", "[dense][dbn][mnist][sgd][ae][momentum]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200>::layer_t,
            dll::rbm_desc<200, 300>::layer_t,
            dll::rbm_desc<300, 28 * 28>::layer_t
        >, dll::momentum, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 50);

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

TEST_CASE("dbn/ae/5", "[dense][dbn][mnist][sgd][ae][momentum][decay]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200>::layer_t,
            dll::rbm_desc<200, 300>::layer_t,
            dll::rbm_desc<300, 28 * 28>::layer_t
        >, dll::momentum, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 50);

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
