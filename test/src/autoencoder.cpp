//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/dense_layer.hpp"
#include "dll/scale_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("dense/ae/1", "[dense][dbn][mnist][sgd][ae]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 200>::layer_t,
            dll::dense_desc<200, 28 * 28>::layer_t
        >, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.1;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 100);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    //auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    //std::cout << "test_error:" << test_error << std::endl;
    //REQUIRE(test_error < 0.2);
}
