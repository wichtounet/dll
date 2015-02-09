//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "catch.hpp"

#define DLL_SVM_SUPPORT

#include "dll/dbn.hpp"
#include "dll/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "dbn/mnist_1", "dbn::simple" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10, 50);

    REQUIRE(error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());

    std::cout << "test_error:" << test_error << std::endl;

    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dbn/mnist_2", "dbn::containers" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::deque, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(200);
    dataset.training_labels.resize(200);

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

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

    typedef dll::dbn_desc<
        dll::dbn_label_layers<
        dll::rbm_desc<28 * 28, 200, dll::batch_size<50>, dll::init_weights, dll::momentum>::rbm_t,
        dll::rbm_desc<200, 300, dll::batch_size<50>, dll::momentum>::rbm_t,
        dll::rbm_desc<310, 500, dll::batch_size<50>, dll::momentum>::rbm_t>>::dbn_t dbn_simple_t;

    auto dbn = std::make_unique<dbn_simple_t>();

    dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 10);

    auto error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::label_predictor());
    REQUIRE(error < 0.3);
}

TEST_CASE( "dbn/mnist_4", "dbn::sgd" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>, dll::trainer<dll::sgd_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100, 10);

    REQUIRE(error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());

    std::cout << "test_error:" << test_error << std::endl;

    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dbn/mnist_5", "dbn::sgd_momentum" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>,
        dll::trainer<dll::sgd_trainer>, dll::momentum>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate *= 2.0;

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100, 10);

    REQUIRE(error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());

    std::cout << "test_error:" << test_error << std::endl;

    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dbn/mnist_6", "dbn::cg_gaussian" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<25>, dll::visible<dll::unit_type::GAUSSIAN>>::rbm_t,
        dll::rbm_desc<200, 500, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<500, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::deque, double>(1000);

    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10, 50);

    REQUIRE(error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());

    std::cout << "test_error:" << test_error << std::endl;

    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dbn/mnist_7", "dbn::sgd_gaussian" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<25>, dll::visible<dll::unit_type::GAUSSIAN>>::rbm_t,
        dll::rbm_desc<200, 500, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<500, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>,
        dll::trainer<dll::sgd_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::deque, double>(1000);

    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.75;

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100, 20);

    REQUIRE(error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());

    std::cout << "test_error:" << test_error << std::endl;

    REQUIRE(test_error < 0.2);
}

//This test should not perform well, but should not fail
TEST_CASE( "dbn/mnist_8", "dbn::cg_relu" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::RELU>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::deque, double>(200);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10, 50);

    REQUIRE(std::isfinite(error));

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());

    std::cout << "test_error:" << test_error << std::endl;
}

//This test should not perform well, but should not fail
TEST_CASE( "dbn/mnist_9", "dbn::sgd_relu" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::RELU>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>,
        dll::trainer<dll::sgd_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::deque, double>(200);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100, 10);

    REQUIRE(std::isfinite(error));
}

TEST_CASE( "dbn/mnist_10", "dbn::sgd_wd" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>,
        dll::trainer<dll::sgd_trainer>, dll::weight_decay<dll::decay_type::L2>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(200);
    dataset.training_labels.resize(200);

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 200, 10);

    REQUIRE(error < 5e-2);
}

TEST_CASE( "dbn/mnist_11", "dbn::sgd_wd_momentum"){
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>,
        dll::trainer<dll::sgd_trainer>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(200);
    dataset.training_labels.resize(200);

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 10);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 200, 10);

    REQUIRE(error < 5e-2);
}

TEST_CASE( "dbn/mnist_12", "dbn::svm_simple" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dbn/mnist_13", "dbn::svm_concatenate" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t>, dll::svm_concatenate>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dbn/mnist_14", "dbn::svm_simple" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dbn/mnist_15", "dbn::parallel" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::parallel, dll::batch_size<25>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::parallel, dll::batch_size<25>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::parallel, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10, 50);

    REQUIRE(error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());

    std::cout << "test_error:" << test_error << std::endl;

    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dbn/mnist_16", "dbn::fast" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<5>, dll::init_weights>::rbm_t,
        dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<5>>::rbm_t,
        dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<5>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(25);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 2, 5);

    REQUIRE(error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());

    std::cout << "test_error:" << test_error << std::endl;

    REQUIRE(test_error < 0.2);
}

//{{{ Performance debugging tests

TEST_CASE( "dbn/mnist_101", "dbn::slow_parallel" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 300, dll::momentum, dll::batch_size<48>, dll::init_weights>::rbm_t,
        dll::rbm_desc<300, 500, dll::momentum, dll::batch_size<48>>::rbm_t,
        dll::rbm_desc<500, 10, dll::momentum, dll::batch_size<48>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1099);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
}

//}}}
