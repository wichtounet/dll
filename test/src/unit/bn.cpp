//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/dense_layer.hpp"
#include "dll/neural/activation_layer.hpp"
#include "dll/neural/batch_normalization_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/datasets.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// (Dense) BN after the non-linearity
TEST_CASE("unit/bn/1", "[unit][bn]") {
    using network_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 200, dll::no_bias, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::batch_normalization_layer_2d_desc<200>::layer_t,

            dll::dense_desc<200, 200, dll::no_bias, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::batch_normalization_layer_2d_desc<200>::layer_t,

            dll::dense_desc<200, 10, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::updater<dll::updater_type::ADADELTA>, dll::batch_size<25>>::dbn_t;

    auto dataset = dll::make_mnist_dataset_val(0, 1000, 2000, 0, dll::batch_size<20>{}, dll::scale_pre<255>{});

    auto dbn = std::make_unique<network_t>();

    dbn->display();
    dataset.display();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK_DATASET_VAL(50, 5e-2);
    TEST_CHECK_DATASET(0.25);
}

// (Dense) BN before the non-linearity
TEST_CASE("unit/bn/2", "[unit][bn]") {
    using network_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_layer_2d_desc<200>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::dense_desc<200, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_layer_2d_desc<200>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::dense_desc<200, 10, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SOFTMAX>::layer_t
        >,
        dll::updater<dll::updater_type::ADADELTA>, dll::batch_size<25>>::dbn_t;

    auto dataset = dll::make_mnist_dataset_val(0, 1000, 3000, 0, dll::batch_size<20>{}, dll::scale_pre<255>{});

    auto dbn = std::make_unique<network_t>();

    dbn->display();
    dataset.display();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK_DATASET_VAL(50, 5e-2);
    TEST_CHECK_DATASET(0.25);
}

// (Conv) BN after the non-linearity
TEST_CASE("unit/bn/3", "[unit][bn]") {
    using network_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 6, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_layer_4d_desc<6, 24, 24>::layer_t,

            dll::conv_desc<6, 24, 24, 6, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_layer_4d_desc<6, 20, 20>::layer_t,

            dll::dense_desc<6 * 20 * 20, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_layer_2d_desc<200>::layer_t,

            dll::dense_desc<200, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_layer_2d_desc<200>::layer_t,

            dll::dense_desc<200, 10, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SOFTMAX>::layer_t
        >,
        dll::updater<dll::updater_type::ADADELTA>, dll::batch_size<25>>::dbn_t;

    auto dataset = dll::make_mnist_dataset_val(0, 500, 2500, 0, dll::batch_size<20>{}, dll::scale_pre<255>{});

    auto dbn = std::make_unique<network_t>();

    dbn->display();
    dataset.display();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK_DATASET_VAL(50, 5e-2);
    TEST_CHECK_DATASET(0.25);
}

// (Conv) BN before the non-linearity
TEST_CASE("unit/bn/4", "[unit][bn]") {
    using network_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 6, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_layer_4d_desc<6, 24, 24>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::conv_desc<6, 24, 24, 6, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_layer_4d_desc<6, 20, 20>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::dense_desc<6 * 20 * 20, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_layer_2d_desc<200>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::dense_desc<200, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_layer_2d_desc<200>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::dense_desc<200, 10, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SOFTMAX>::layer_t
        >,
        dll::updater<dll::updater_type::ADADELTA>, dll::batch_size<25>>::dbn_t;

    auto dataset = dll::make_mnist_dataset_val(0, 500, 2500, 0, dll::batch_size<20>{}, dll::scale_pre<255>{});

    auto dbn = std::make_unique<network_t>();

    dbn->display();
    dataset.display();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK_DATASET_VAL(50, 5e-2);
    TEST_CHECK_DATASET(0.25);
}

// (Conv+MP) BN after the non-linearity
TEST_CASE("unit/bn/5", "[unit][bn]") {
    constexpr size_t K = 6;

    using network_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, K, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::mp_layer_2d_desc<K, 24, 24, 2, 2>::layer_t,
            dll::batch_normalization_layer_4d_desc<K, 12, 12>::layer_t,

            dll::conv_desc<K, 12, 12, K, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::mp_layer_2d_desc<K, 8, 8, 2, 2>::layer_t,
            dll::batch_normalization_layer_4d_desc<K, 4, 4>::layer_t,

            dll::dense_desc<K * 4 * 4, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_layer_2d_desc<200>::layer_t,

            dll::dense_desc<200, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_layer_2d_desc<200>::layer_t,

            dll::dense_desc<200, 10, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SOFTMAX>::layer_t
        >,
        dll::updater<dll::updater_type::ADADELTA>, dll::batch_size<25>>::dbn_t;

    auto dataset = dll::make_mnist_dataset_val(0, 500, 2500, 0, dll::batch_size<20>{}, dll::scale_pre<255>{});

    auto dbn = std::make_unique<network_t>();

    dbn->display();
    dataset.display();

    dbn->initial_momentum = 0.9;
    dbn->final_momentum   = 0.9;
    dbn->learning_rate    = 0.01;

    FT_CHECK_DATASET_VAL(50, 5e-2);
    TEST_CHECK_DATASET(0.25);
}
