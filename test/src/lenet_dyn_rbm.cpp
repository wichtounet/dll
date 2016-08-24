//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "dll/dyn_rbm.hpp"
#include "dll/dyn_conv_rbm.hpp"
#include "dll/scale_layer.hpp"
#include "dll/dyn_mp_layer.hpp"
#include "dll/dyn_avgp_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("dyn_lenet_rbm", "[dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::scale_layer_desc<1, 256>::layer_t,
            dll::dyn_conv_rbm_desc<dll::hidden<dll::unit_type::RELU>, dll::momentum, dll::weight_type<float>>::layer_t,
            dll::dyn_mp_layer_3d_desc<dll::weight_type<float>>::layer_t,
            dll::dyn_conv_rbm_desc<dll::hidden<dll::unit_type::RELU>, dll::momentum, dll::weight_type<float>>::layer_t,
            dll::dyn_mp_layer_3d_desc<dll::weight_type<float>>::layer_t,
            dll::dyn_rbm_desc<dll::hidden<dll::unit_type::BINARY>, dll::momentum>::layer_t,
            dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::momentum, dll::weight_decay<>, dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<1>().init_layer(1, 28, 28, 20, 24, 24);
    dbn->template layer_get<2>().init_layer(20, 24, 24, 1, 2, 2);
    dbn->template layer_get<3>().init_layer(20, 12, 12, 50, 8, 8);
    dbn->template layer_get<4>().init_layer(50, 8, 8, 1, 2, 2);
    dbn->template layer_get<5>().init_layer(50 * 4 * 4, 500);
    dbn->template layer_get<6>().init_layer(500, 10);

    dbn->l2_weight_cost   = 0.0005;
    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.1;

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;
    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}
