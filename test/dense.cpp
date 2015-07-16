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

TEST_CASE( "dense/sgd/1", "[dense][dbn][mnist][sgd]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 100>::layer_t,
            dll::dense_desc<100, 10>::layer_t
        >,
        dll::trainer<dll::dense_sgd_trainer>
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);

    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100, 10);
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
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);

    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100, 10);
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
    >::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);

    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum = 0.9;
    dbn->learning_rate = 0.01;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 100, 10);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}
