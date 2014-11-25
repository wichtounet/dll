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

#include "dll/dbn.hpp"
#include "dll/dbn_desc.hpp"

#include "dll/dyn_dbn_desc.hpp"

#include "dll/dbn_layers.hpp"
#include "dll/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "dyn_dbn/mnist_1", "dbn::simple" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::dyn_rbm_desc<dll::momentum, dll::init_weights>::rbm_t,
        dll::dyn_rbm_desc<dll::momentum>::rbm_t,
        dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>>::dbn_t dbn_t;

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
