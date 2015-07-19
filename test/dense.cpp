//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "catch.hpp"

#include "dll/dense_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/dense_stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

template<typename Dataset>
void mnist_scale(Dataset& dataset){
    for(auto& image : dataset.training_images){
        for(auto& pixel : image){
            pixel *= (1.0 / 256.0);
        }
    }

    for(auto& image : dataset.test_images){
        for(auto& pixel : image){
            pixel *= (1.0 / 256.0);
        }
    }
}

} //end of anonymous namespace

TEST_CASE( "dense/sgd/1", "[dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100>::layer_t,
            dll::dense_desc<100, 10>::layer_t
        >,
        dll::trainer<dll::dense_sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dense/sgd/2", "[dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100, dll::activation<dll::function::TANH>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::TANH>>::layer_t
        >,
        dll::trainer<dll::dense_sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);
    REQUIRE(!dataset.training_images.empty());

    mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dense/sgd/3", "[dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100>::layer_t,
            dll::dense_desc<100, 10>::layer_t
        >
        , dll::momentum
        , dll::trainer<dll::dense_sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);
    REQUIRE(!dataset.training_images.empty());

    mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dense/sgd/4", "[dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100>::layer_t,
            dll::dense_desc<100, 10>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::dense_sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);
    REQUIRE(!dataset.training_images.empty());

    mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dense/sgd/5", "[dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100, dll::activation<dll::function::TANH>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::TANH>>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::dense_sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);
    REQUIRE(!dataset.training_images.empty());

    mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "dense/sgd/6", "[dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100, dll::activation<dll::function::IDENTITY>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::IDENTITY>>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::dense_sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);
    REQUIRE(!dataset.training_images.empty());

    mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.4);
}

TEST_CASE( "dense/sgd/7", "[dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<100, 10, dll::activation<dll::function::SIGMOID>>::layer_t
        >
        , dll::momentum
        , dll::weight_decay<>
        , dll::trainer<dll::dense_sgd_trainer>
        , dll::batch_size<10>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);
    REQUIRE(!dataset.training_images.empty());

    mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.4);
}
