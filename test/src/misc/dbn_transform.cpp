//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

//Test for DBN and transform layers

#include <deque>

#include "dll_test.hpp"

#include "dll/transform/shape_1d_layer.hpp"
#include "dll/transform/binarize_layer.hpp"
#include "dll/transform/normalize_layer.hpp"
#include "dll/rbm/rbm.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dbn/mnist_18", "binarize_layer_impl") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::shape_1d_layer_desc<28 * 28>::layer_t,
            dll::binarize_layer_desc<30>::layer_t,
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
}

DLL_TEST_CASE("dbn/mnist_19", "normalize_layer_impl") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::shape_1d_layer_desc<28 * 28>::layer_t,
            dll::normalize_layer_desc::layer_t,
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<25>, dll::visible<dll::unit_type::GAUSSIAN>>::layer_t,
            dll::rbm_desc<200, 500, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<500, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1000);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
}
