//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "catch.hpp"

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/conjugate_gradient.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("hybrid/mnist/1", "[hybrid]") {
    typedef dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<50>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10);
    std::cout << "ft_error:" << ft_error << std::endl;
    REQUIRE(ft_error < 5e-2);

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}
