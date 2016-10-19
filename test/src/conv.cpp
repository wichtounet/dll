//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/conv_layer.hpp"
#include "dll/dense_layer.hpp"
#include "dll/transform/scale_layer.hpp"
#include "dll/mp_layer.hpp"
#include "dll/avgp_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("conv/sgd/1", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 24, 24, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_desc<10 * 24 * 24, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE("conv/sgd/2", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 24, 24, dll::activation<dll::function::TANH>>::layer_t,
            dll::dense_desc<10 * 24 * 24, 10, dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE("conv/sgd/3", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 24, 24, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<10 * 24 * 24, 10, dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE("conv/sgd/4", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 24, 24, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::conv_desc<10, 24, 24, 6, 20, 20, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_desc<6 * 20 * 20, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE("conv/sgd/5", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 24, 24, dll::activation<dll::function::RELU>>::layer_t,
            dll::conv_desc<10, 24, 24, 6, 20, 20, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<6 * 20 * 20, 200, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<200, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE("conv/sgd/6", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 24, 24, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<10, 24, 24, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_desc<10, 12, 12, 6, 8, 8, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<6 * 8 * 8, 100, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE("conv/sgd/7", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 24, 24, dll::activation<dll::function::RELU>>::layer_t,
            dll::avgp_layer_3d_desc<10, 24, 24, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_desc<10, 12, 12, 6, 8, 8, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<6 * 8 * 8, 100, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE("lenet", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 20, 24, 24, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<20, 24, 24, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_desc<20, 12, 12, 50, 8, 8, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<50, 8, 8, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::dense_desc<50 * 4 * 4, 500, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<500, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::momentum, dll::weight_decay<>, dll::trainer<dll::sgd_trainer>, dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->l2_weight_cost   = 0.0005;
    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE("conv/sgd/8", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::scale_layer_desc<1, 256>::layer_t,
            dll::conv_desc<1, 28, 28, 10, 24, 24, dll::activation<dll::function::TANH>>::layer_t,
            dll::dense_desc<10 * 24 * 24, 10, dll::activation<dll::function::TANH>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}
