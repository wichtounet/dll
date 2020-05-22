//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/conv/dyn_conv_layer.hpp"
#include "dll/neural/dense/dyn_dense_layer.hpp"
#include "dll/pooling/dyn_mp_layer.hpp"
#include "dll/pooling/dyn_avgp_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dyn/conv/sgd/1", "[dense][dbn][mnist][sgd]") {
    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_conv_layer_desc<dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(1, 28, 28, 10, 5, 5);
    dbn->template layer_get<1>().init_layer(10 * 24 * 24, 10);

    dbn->learning_rate = 0.05;

    dbn->display();

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("dyn/conv/sgd/2", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(1, 28, 28, 10, 5, 5);
    dbn->template layer_get<1>().init_layer(10, 24, 24, 6, 5, 5);
    dbn->template layer_get<2>().init_layer(6 * 20 * 20, 200);
    dbn->template layer_get<3>().init_layer(200, 10);

    dbn->learning_rate = 0.05;

    dbn->display();

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("dyn/conv/sgd/3", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_mp_3d_layer_desc<dll::weight_type<float>>::layer_t,
            dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(1, 28, 28, 10, 5, 5);
    dbn->template layer_get<1>().init_layer(10, 24, 24, 1, 2, 2 );
    dbn->template layer_get<2>().init_layer(10, 12, 12, 6, 5, 5);
    dbn->template layer_get<3>().init_layer(6 * 8 * 8, 100);
    dbn->template layer_get<4>().init_layer(100, 10);

    dbn->learning_rate = 0.05;

    dbn->display();

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("dyn/conv/sgd/4", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_avgp_3d_layer_desc<dll::weight_type<float>>::layer_t,
            dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
            dll::dyn_dense_layer_desc<dll::activation<dll::function::SIGMOID>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(1000);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(1, 28, 28, 10, 5, 5);
    dbn->template layer_get<1>().init_layer(10, 24, 24, 1, 2, 2 );
    dbn->template layer_get<2>().init_layer(10, 12, 12, 6, 5, 5);
    dbn->template layer_get<3>().init_layer(6 * 8 * 8, 100);
    dbn->template layer_get<4>().init_layer(100, 10);

    dbn->learning_rate = 0.05;

    dbn->display();

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}
