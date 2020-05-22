//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/conv/conv_layer.hpp"
#include "dll/neural/dense/dense_layer.hpp"
#include "dll/neural/activation/activation_layer.hpp"
#include "dll/neural/bn/batch_normalization_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// (Dense) BN after the non-linearity
DLL_TEST_CASE("unit/bn/1", "[unit][bn]") {
    using network_t = dll::network_desc<
        dll::network_layers<
            dll::dense_layer_desc<28 * 28, 200, dll::no_bias, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::batch_normalization_2d_layer_desc<200>::layer_t,

            dll::dense_layer_desc<200, 200, dll::no_bias, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::batch_normalization_2d_layer_desc<200>::layer_t,

            dll::dense_layer_desc<200, 10, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::updater<dll::updater_type::ADADELTA>, dll::early_training, dll::batch_size<25>>::network_t;

    auto dataset = dll::make_mnist_dataset_val(0, 1000, 2000, dll::batch_size<25>{}, dll::scale_pre<255>{});

    auto net = std::make_unique<network_t>();

    net->display();
    dataset.display();

    net->initial_momentum = 0.9;
    net->final_momentum   = 0.9;
    net->learning_rate    = 0.01;

    FT_CHECK_2_VAL(net, dataset, 50, 5e-2);
    TEST_CHECK_2(net, dataset, 0.25);
}

// (Dense) BN before the non-linearity
DLL_TEST_CASE("unit/bn/2", "[unit][bn]") {
    using network_t = dll::network_desc<
        dll::network_layers<
            dll::dense_layer_desc<28 * 28, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_2d_layer_desc<200>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::dense_layer_desc<200, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_2d_layer_desc<200>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::dense_layer_desc<200, 10, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SOFTMAX>::layer_t
        >,
        dll::updater<dll::updater_type::ADADELTA>, dll::early_training, dll::batch_size<25>>::network_t;

    auto dataset = dll::make_mnist_dataset_val(0, 1000, 3000, dll::batch_size<25>{}, dll::scale_pre<255>{});

    auto net = std::make_unique<network_t>();

    net->display();
    dataset.display();

    net->initial_momentum = 0.9;
    net->final_momentum   = 0.9;
    net->learning_rate    = 0.01;

    FT_CHECK_2_VAL(net, dataset, 50, 5e-2);
    TEST_CHECK_2(net, dataset, 0.25);
}

// (Conv) BN after the non-linearity
DLL_TEST_CASE("unit/bn/3", "[unit][bn]") {
    using network_t = dll::network_desc<
        dll::network_layers<
            dll::conv_layer_desc<1, 28, 28, 6, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_4d_layer_desc<6, 24, 24>::layer_t,

            dll::conv_layer_desc<6, 24, 24, 6, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_4d_layer_desc<6, 20, 20>::layer_t,

            dll::dense_layer_desc<6 * 20 * 20, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_2d_layer_desc<200>::layer_t,

            dll::dense_layer_desc<200, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_2d_layer_desc<200>::layer_t,

            dll::dense_layer_desc<200, 10, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SOFTMAX>::layer_t
        >,
        dll::updater<dll::updater_type::ADADELTA>, dll::early_training, dll::batch_size<25>>::network_t;

    auto dataset = dll::make_mnist_dataset_val(0, 500, 2500, dll::batch_size<25>{}, dll::scale_pre<255>{});

    auto net = std::make_unique<network_t>();

    net->display();
    dataset.display();

    net->initial_momentum = 0.9;
    net->final_momentum   = 0.9;
    net->learning_rate    = 0.01;

    FT_CHECK_2_VAL(net, dataset, 50, 5e-2);
    TEST_CHECK_2(net, dataset, 0.25);
}

// (Conv) BN before the non-linearity
DLL_TEST_CASE("unit/bn/4", "[unit][bn]") {
    using network_t = dll::network_desc<
        dll::network_layers<
            dll::conv_layer_desc<1, 28, 28, 6, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_4d_layer_desc<6, 24, 24>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::conv_layer_desc<6, 24, 24, 6, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_4d_layer_desc<6, 20, 20>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::dense_layer_desc<6 * 20 * 20, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_2d_layer_desc<200>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::dense_layer_desc<200, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::batch_normalization_2d_layer_desc<200>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,

            dll::dense_layer_desc<200, 10, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SOFTMAX>::layer_t
        >,
        dll::updater<dll::updater_type::ADADELTA>, dll::early_training, dll::batch_size<25>>::network_t;

    auto dataset = dll::make_mnist_dataset_val(0, 500, 2500, dll::batch_size<25>{}, dll::scale_pre<255>{});

    auto net = std::make_unique<network_t>();

    net->display();
    dataset.display();

    net->initial_momentum = 0.9;
    net->final_momentum   = 0.9;
    net->learning_rate    = 0.01;

    FT_CHECK_2_VAL(net, dataset, 50, 5e-2);
    TEST_CHECK_2(net, dataset, 0.25);
}

// (Conv+MP) BN after the non-linearity
DLL_TEST_CASE("unit/bn/5", "[unit][bn]") {
    constexpr size_t K = 6;

    using network_t = dll::network_desc<
        dll::network_layers<
            dll::conv_layer_desc<1, 28, 28, K, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::mp_2d_layer_desc<K, 24, 24, 2, 2>::layer_t,
            dll::batch_normalization_4d_layer_desc<K, 12, 12>::layer_t,

            dll::conv_layer_desc<K, 12, 12, K, 5, 5, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::mp_2d_layer_desc<K, 8, 8, 2, 2>::layer_t,
            dll::batch_normalization_4d_layer_desc<K, 4, 4>::layer_t,

            dll::dense_layer_desc<K * 4 * 4, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_2d_layer_desc<200>::layer_t,

            dll::dense_layer_desc<200, 200, dll::no_bias, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::batch_normalization_2d_layer_desc<200>::layer_t,

            dll::dense_layer_desc<200, 10, dll::no_activation>::layer_t,
            dll::activation_layer_desc<dll::function::SOFTMAX>::layer_t
        >,
        dll::updater<dll::updater_type::ADADELTA>, dll::early_training, dll::batch_size<25>>::network_t;

    auto dataset = dll::make_mnist_dataset_val(0, 500, 2500, dll::batch_size<25>{}, dll::scale_pre<255>{});

    auto net = std::make_unique<network_t>();

    net->display();
    dataset.display();

    net->initial_momentum = 0.9;
    net->final_momentum   = 0.9;
    net->learning_rate    = 0.01;

    FT_CHECK_2_VAL(net, dataset, 50, 5e-2);
    TEST_CHECK_2(net, dataset, 0.25);
}
