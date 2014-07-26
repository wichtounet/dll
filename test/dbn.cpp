//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "catch.hpp"

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "dbn/mnist_1", "dbn::simple" ) {
    typedef dll::dbn<
        dll::layer<28 * 28, 100, dll::in_dbn, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::layer<100, 200, dll::in_dbn, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::layer<200, 10, dll::in_dbn, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t> dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(200);
    dataset.training_labels.resize(200);

    mnist::binarize_dataset(dataset);

    auto dbn = make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 5, 50);

    REQUIRE(error < 5e-2);
}

TEST_CASE( "dbn/mnist_2", "dbn::containers" ) {
    typedef dll::dbn<
        dll::layer<28 * 28, 100, dll::in_dbn, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::layer<100, 200, dll::in_dbn, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::layer<200, 10, dll::in_dbn, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t> dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::deque, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(200);
    dataset.training_labels.resize(200);

    mnist::binarize_dataset(dataset);

    auto dbn = make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 5, 50);

    REQUIRE(error < 5e-2);
}

TEST_CASE( "dbn/mnist_3", "dbn::labels" ) {
    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(1000);
    dataset.training_labels.resize(1000);

    mnist::binarize_dataset(dataset);

    typedef dll::dbn<
        dll::layer<28 * 28, 200, dll::in_dbn, dll::batch_size<50>, dll::init_weights, dll::momentum>::rbm_t,
        dll::layer<200, 300, dll::in_dbn, dll::batch_size<50>, dll::momentum>::rbm_t,
        dll::layer<310, 500, dll::in_dbn, dll::batch_size<50>, dll::momentum>::rbm_t> dbn_simple_t;

    auto dbn = make_unique<dbn_simple_t>();

    dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 10);

    auto error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::label_predictor());
    REQUIRE(error < 0.3);
}