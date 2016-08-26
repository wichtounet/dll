//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"
#include "dll_test.hpp"

#include "dll/mp_layer.hpp"
#include "dll/avgp_layer.hpp"
#include "dll/dense_layer.hpp"
#include "dll/conv_layer.hpp"
#include "dll/random_layer.hpp"
#include "dll/binarize_layer.hpp"
#include "dll/normalize_layer.hpp"
#include "dll/rbm.hpp"
#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/conjugate_gradient.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"
#include "dll/scale_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("hybrid/mnist/1", "[hybrid]") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<50>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10);
    std::cout << "ft_error:" << ft_error << std::endl;
    REQUIRE(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE("hybrid/mnist/2", "") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 10, 24, 24, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<10, 24, 24, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_desc<10, 12, 12, 6, 8, 8, dll::activation<dll::function::RELU>>::layer_t,
            dll::avgp_layer_3d_desc<6, 8, 8, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::dense_desc<6 * 4 * 4, 100, dll::activation<dll::function::RELU>>::layer_t,
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

TEST_CASE("hybrid/mnist/3", "") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_mp_desc_square<1, 28, 40, 12, 2, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_mp_desc_square<40, 6, 20, 4, 2, dll::momentum, dll::batch_size<25>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
}

TEST_CASE("hybrid/mnist/4", "") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc_square<1, 28, 40, 12, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_desc_square<40, 12, 20, 10, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_desc_square<20, 10, 50, 6, dll::momentum, dll::batch_size<25>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
}

TEST_CASE("hybrid/mnist/5", "[cdbn][rectifier][svm][unit]") {
    using dbn_t =
        dll::dyn_dbn_desc<dll::dbn_layers<
              dll::conv_rbm_desc_square<1, 28, 20, 12, dll::parallel_mode, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::random_layer_desc::layer_t
            , dll::conv_rbm_desc_square<20, 12, 20, 10, dll::parallel_mode, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();
    dbn->display();
}

TEST_CASE("hybrid/mnist/6", "") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::binarize_layer_desc<30>::layer_t,
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(100);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
}

TEST_CASE("hybrid/mnist/7", "") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::normalize_layer_desc::layer_t,
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<25>, dll::visible<dll::unit_type::GAUSSIAN>>::layer_t,
            dll::rbm_desc<200, 500, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<500, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(100);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
}
