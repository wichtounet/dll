//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "catch.hpp"

#define DLL_SVM_SUPPORT

#include "dll/dyn_rbm.hpp"
#include "dll/dyn_dbn.hpp"

#include "dll/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "dyn_dbn/mnist_1", "dbn::simple" ) {
    using dbn_t =
        dll::dyn_dbn_desc<
            dll::dbn_dyn_layers<
                dll::dyn_rbm_desc<dll::momentum, dll::init_weights>::rbm_t,
                dll::dyn_rbm_desc<dll::momentum>::rbm_t,
                dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>(
        std::make_tuple(28*28,100),
        std::make_tuple(100,200),
        std::make_tuple(200,10));

    dbn->pretrain(dataset.training_images, 20);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());

    std::cout << "test_error:" << test_error << std::endl;

    REQUIRE(test_error < 1.0);
}

TEST_CASE( "dyn_dbn/mnist_2", "dbn::parallel" ) {
    using dbn_t =
        dll::dyn_dbn_desc<
            dll::dbn_dyn_layers<
                dll::dyn_rbm_desc<dll::momentum, dll::parallel, dll::init_weights>::rbm_t,
                dll::dyn_rbm_desc<dll::momentum, dll::parallel>::rbm_t,
                dll::dyn_rbm_desc<dll::momentum, dll::parallel, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>(
        std::make_tuple(28*28,100),
        std::make_tuple(100,200),
        std::make_tuple(200,10));

    dbn->pretrain(dataset.training_images, 20);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());

    std::cout << "test_error:" << test_error << std::endl;

    REQUIRE(test_error < 1.0);
}

TEST_CASE( "dyn_dbn/mnist_3", "dbn::labels" ) {
    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(1000);
    dataset.training_labels.resize(1000);

    mnist::binarize_dataset(dataset);

    using dbn_t =
        dll::dyn_dbn_desc<
            dll::dbn_dyn_layers<
                dll::dyn_rbm_desc<dll::init_weights, dll::momentum>::rbm_t,
                dll::dyn_rbm_desc<dll::momentum>::rbm_t,
                dll::dyn_rbm_desc<dll::momentum>::rbm_t
        >>::dbn_t;

    auto dbn = std::make_unique<dbn_t>(
        std::make_tuple(28*28,200),
        std::make_tuple(200,300),
        std::make_tuple(310,500));

    dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 10);

    auto error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::label_predictor());
    REQUIRE(error < 0.3);
}
