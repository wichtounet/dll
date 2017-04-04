//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/dense_layer.hpp"
#include "dll/neural/activation_layer.hpp"
#include "dll/transform/scale_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "cifar/cifar10_reader.hpp"

// Fully-Connected Network on CIFAR-10

TEST_CASE("cifar/dense/sgd/1", "[dense][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<3 * 32 * 32, 1000>::layer_t,
            dll::dense_desc<1000, 500>::layer_t,
            dll::dense_desc<500, 10>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::momentum, dll::batch_size<20>>::dbn_t dbn_t;

    auto dataset = cifar::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 3 * 32 * 32>>(2000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->learning_rate = 0.01;
    dbn->momentum = 0.9;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}
