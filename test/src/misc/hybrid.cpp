//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/dbn.hpp"
#include "dll/neural/conv/conv_layer.hpp"
#include "dll/neural/dense/dense_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/pooling/avgp_layer.hpp"
#include "dll/rbm/rbm.hpp"
#include "dll/rbm/conv_rbm.hpp"
#include "dll/rbm/conv_rbm_mp.hpp"

#include "dll/transform/random_layer.hpp"
#include "dll/transform/binarize_layer.hpp"
#include "dll/transform/normalize_layer.hpp"
#include "dll/transform/rectifier_layer.hpp"
#include "dll/transform/lcn_layer.hpp"
#include "dll/transform/shape_1d_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("hybrid/mnist/1", "[hybrid]") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<50>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10);
    std::cout << "ft_error:" << ft_error << std::endl;
    REQUIRE(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("hybrid/mnist/2", "") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 10, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_3d_layer_desc<10, 24, 24, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_layer_desc<10, 12, 12, 6, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::avgp_3d_layer_desc<6, 8, 8, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::dense_layer_desc<6 * 4 * 4, 100, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("hybrid/mnist/3", "") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_mp_desc_square<1, 28, 40, 17, 2, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_mp_desc_square<40, 6, 20, 3, 2, dll::momentum, dll::batch_size<25>>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
}

DLL_TEST_CASE("hybrid/mnist/4", "") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_square_desc<1, 28, 40, 17, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_square_desc<40, 12, 20, 3, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_square_desc<20, 10, 50, 5, dll::momentum, dll::batch_size<25>>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
}

DLL_TEST_CASE("hybrid/mnist/6", "") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::shape_1d_layer_desc<28 * 28>::layer_t,
            dll::binarize_layer_desc<30>::layer_t,
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(100);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
}

DLL_TEST_CASE("hybrid/mnist/7", "") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::shape_1d_layer_desc<28 * 28>::layer_t,
            dll::normalize_layer_desc::layer_t,
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<25>, dll::visible<dll::unit_type::GAUSSIAN>>::layer_t,
            dll::rbm_desc<200, 500, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<500, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(100);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
}

DLL_TEST_CASE("hybrid/mnist/8", "[dense][dbn][mnist][sgd]") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::weight_decay<>, dll::scale_pre<255>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(350);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

DLL_TEST_CASE("hybrid/mnist/10", "") {
    using dbn_t =
        dll::dyn_dbn_desc<dll::dbn_layers<
              dll::conv_rbm_square_desc<1, 28, 20, 17, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::conv_rbm_square_desc<20, 12, 20, 3, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::lcn_layer_desc<9>::layer_t
        >, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();
    dbn->display();
}
