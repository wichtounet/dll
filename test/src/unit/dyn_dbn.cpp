//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#define DLL_SVM_SUPPORT

#include "dll/rbm/dyn_rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/transform/shape_1d_layer.hpp"
#include "dll/transform/binarize_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("unit/dyn_dbn/mnist/1", "[dyn_dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_rbm_desc<dll::momentum, dll::init_weights>::layer_t,
            dll::dyn_rbm_desc<dll::momentum>::layer_t,
            dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<25>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(350);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 150);
    dbn->template layer_get<1>().init_layer(150, 150);
    dbn->template layer_get<2>().init_layer(150, 10);

    dbn->pretrain(dataset.training_images, 50);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 5);
    std::cout << "ft_error:" << ft_error << std::endl;
    REQUIRE(ft_error < 5e-2);

    TEST_CHECK(0.25);
}

DLL_TEST_CASE("unit/dyn_dbn/mnist/2", "[dyn_dbn][sgd][unit]") {
    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::shape_1d_layer_desc<28 * 28>::layer_t,
            dll::binarize_layer_desc<30>::layer_t,
            dll::dyn_rbm_desc<dll::momentum, dll::init_weights>::layer_t,
            dll::dyn_rbm_desc<dll::momentum>::layer_t,
            dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<25>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(250);

    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<2>().init_layer(28 * 28, 150);
    dbn->template layer_get<3>().init_layer(150, 200);
    dbn->template layer_get<4>().init_layer(200, 10);

    dbn->learning_rate = 0.05;

    dbn->pretrain(dataset.training_images, 20);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;
    REQUIRE(ft_error < 1e-1);

    TEST_CHECK(0.3);
}
