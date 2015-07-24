//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "catch.hpp"

#define DLL_SVM_SUPPORT

#include "dll/dyn_rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/binarize_layer.hpp"
#include "dll/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "unit/dbn/mnist/1", "[dbn][unit]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 125, dll::momentum, dll::batch_size<10>, dll::init_weights>::rbm_t,
            dll::rbm_desc<125, 250, dll::momentum, dll::batch_size<10>>::rbm_t,
            dll::rbm_desc<250, 10, dll::momentum, dll::batch_size<10>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>
        , dll::batch_size<10>
        >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 5);
    REQUIRE(error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "unit/dbn/mnist/2", "[dbn][unit]" ) {
    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    typedef dll::dbn_desc<
        dll::dbn_label_layers<
            dll::rbm_desc<28 * 28, 200, dll::batch_size<50>, dll::init_weights, dll::momentum>::rbm_t,
            dll::rbm_desc<200, 300, dll::batch_size<50>, dll::momentum>::rbm_t,
            dll::rbm_desc<310, 500, dll::batch_size<50>, dll::momentum>::rbm_t>
        , dll::batch_size<10>
        >::dbn_t dbn_simple_t;

    auto dbn = std::make_unique<dbn_simple_t>();

    dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 10);

    auto error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::label_predictor());
    std::cout << "test_error:" << error << std::endl;
    REQUIRE(error < 0.3);
}

TEST_CASE( "unit/dbn/mnist/3", "[dbn][unit]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<20>, dll::visible<dll::unit_type::GAUSSIAN>>::rbm_t,
            dll::rbm_desc<200, 350, dll::momentum, dll::batch_size<20>>::rbm_t,
            dll::rbm_desc<350, 10, dll::momentum, dll::batch_size<20>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>
        , dll::batch_size<10>
        >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::deque, double>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 5);
    REQUIRE(error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "unit/dbn/mnist/4", "[dbn][cg][unit]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 150, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
            dll::rbm_desc<150, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>
        , dll::memory
        , dll::batch_size<25>
        >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto error = dbn->fine_tune(
        dataset.training_images.begin(), dataset.training_images.end(),
        dataset.training_labels.begin(), dataset.training_labels.end(),
        5);

    REQUIRE(error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    REQUIRE(test_error < 0.25);

    //Mostly here to ensure compilation
    auto out = dbn->prepare_one_output<std::vector<double>>();
    REQUIRE(out.size() > 0);
}

TEST_CASE( "unit/dbn/mnist/5", "[dbn][sgd][unit]" ) {
    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 150, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
            dll::rbm_desc<150, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>
        , dll::trainer<dll::sgd_trainer>
        , dll::momentum
        , dll::batch_size<25>
        >::dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    dbn->pretrain(dataset.training_images, 20);

    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << error << std::endl;
    REQUIRE(error < 1e-1);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << error << std::endl;
    REQUIRE(test_error < 0.3);
}

TEST_CASE( "unit/dbn/mnist/6", "[dbn][dyn][unit]" ) {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_rbm_desc<dll::momentum, dll::init_weights>::rbm_t,
                dll::dyn_rbm_desc<dll::momentum>::rbm_t,
                dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>
        , dll::batch_size<25>
        >::dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(250);

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

TEST_CASE( "unit/dbn/mnist/7", "[dbn][svm][unit]" ) {
    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t>
        , dll::batch_size<25>
        >::dbn_t;

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

TEST_CASE( "unit/dbn/mnist/8", "[dbn][unit]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::binarize_layer_desc<30>::layer_t,
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>
        , dll::batch_size<25>
        >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(250);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();
    dbn->pretrain(dataset.training_images, 20);
}
