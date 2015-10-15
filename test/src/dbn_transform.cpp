//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

//Test for DBN and transform layers

#include <deque>

#include "catch.hpp"

#include "dll/dbn.hpp"

#include "dll/binarize_layer.hpp"
#include "dll/normalize_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("dbn/mnist_18", "binarize_layer") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::binarize_layer_desc<30>::layer_t,
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::rbm_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::rbm_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
}

TEST_CASE("dbn/mnist_19", "normalize_layer") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::normalize_layer_desc::layer_t,
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<25>, dll::visible<dll::unit_type::GAUSSIAN>>::rbm_t,
            dll::rbm_desc<200, 500, dll::momentum, dll::batch_size<25>>::rbm_t,
            dll::rbm_desc<500, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
}
