//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/conv/conv_same_layer.hpp"
#include "dll/neural/conv/conv_layer.hpp"
#include "dll/neural/dense/dense_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/pooling/mp_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("unit/conv/same/1", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_same_desc<1, 28, 28, 6, 3, 3, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_layer_desc<6 * 28 * 28, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<20>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(600);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    dbn->display();

    FT_CHECK(50, 5e-2);
    TEST_CHECK(0.2);
}

DLL_TEST_CASE("unit/conv/same/2", "[conv][dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_same_desc<1, 28, 28, 6, 3, 3, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::mp_3d_layer_desc<6, 28, 28, 1, 2, 2>::layer_t,
            dll::dense_layer_desc<6 * 14 * 14, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<20>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(600);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    dbn->display();

    FT_CHECK(100, 5e-2);
    TEST_CHECK(0.2);
}
