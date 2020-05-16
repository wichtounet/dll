//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#define DLL_SVM_SUPPORT

#include "dll/rbm/dyn_rbm.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dyn_dbn/mnist_1", "dbn::simple") {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_rbm_desc<dll::momentum, dll::init_weights>::layer_t,
                dll::dyn_rbm_desc<dll::momentum>::layer_t,
                dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 200);
    dbn->template layer_get<2>().init_layer(200, 10);

    dbn->pretrain(dataset.training_images, 20);

    TEST_CHECK(1.0);
}

DLL_TEST_CASE("dyn_dbn/mnist_3", "dbn::labels") {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_rbm_desc<dll::init_weights, dll::momentum>::layer_t,
                dll::dyn_rbm_desc<dll::momentum>::layer_t,
                dll::dyn_rbm_desc<dll::momentum>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1000);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 200);
    dbn->template layer_get<1>().init_layer(200, 300);
    dbn->template layer_get<2>().init_layer(310, 500);

    dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 10);

    auto error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::label_predictor());
    REQUIRE(error < 0.3);
}

DLL_TEST_CASE("dyn_dbn/mnist_4", "dbn::svm_simple") {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_rbm_desc<dll::momentum, dll::init_weights>::layer_t,
                dll::dyn_rbm_desc<dll::momentum>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 150);
    dbn->template layer_get<1>().init_layer(150, 250);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    dbn->pretrain(dataset.training_images, 20);
    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

//This test is mostly here to ensure compilation

DLL_TEST_CASE("dyn_dbn/mnist_5", "dbn::simple_single") {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_rbm_desc<dll::momentum, dll::init_weights>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 100);

    dbn->pretrain(dataset.training_images, 20);
}

//This test is here for debugging purposes
DLL_TEST_CASE("dyn_dbn/mnist_6", "dbn::labels_fast") {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(25, 25);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_rbm_desc<dll::init_weights, dll::momentum>::layer_t,
                dll::dyn_rbm_desc<dll::momentum>::layer_t,
                dll::dyn_rbm_desc<dll::momentum>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 80);
    dbn->template layer_get<1>().init_layer(80, 100);
    dbn->template layer_get<2>().init_layer(110, 130);

    dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 5);

    auto error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::label_predictor());
    REQUIRE(error < 1.0);
}
