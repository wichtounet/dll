//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/neural/rnn/rnn_layer.hpp"
#include "dll/neural/recurrent/recurrent_last_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

// Simple RNN
DLL_TEST_CASE("unit/rnn/1", "[unit][rnn]") {
    auto dataset = dll::make_mnist_dataset_nc_sub(0, 2000, dll::batch_size<100>{}, dll::scale_pre<255>{});

    constexpr size_t time_steps      = 28;
    constexpr size_t sequence_length = 28;
    constexpr size_t hidden_units    = 75;

    using network_t = dll::dyn_network_desc<
        dll::network_layers<
            dll::rnn_layer<time_steps, sequence_length, hidden_units, dll::last_only>,
            dll::recurrent_last_layer<time_steps, hidden_units>,
            dll::dense_layer<hidden_units, 10, dll::softmax>
        >
        , dll::updater<dll::updater_type::ADAM>      // Adam
        , dll::batch_size<100>                       // The mini-batch size
    >::network_t;

    auto net = std::make_unique<network_t>();

    REQUIRE(net->fine_tune(dataset.train(), 30) < 0.15);
    REQUIRE(net->evaluate_error(dataset.test()) < 0.25);
}

// Simple RNN with truncation
DLL_TEST_CASE("unit/rnn/2", "[unit][rnn]") {
    auto dataset = dll::make_mnist_dataset_nc_sub(0, 2000, dll::batch_size<100>{}, dll::scale_pre<255>{});

    constexpr size_t time_steps      = 28;
    constexpr size_t sequence_length = 28;
    constexpr size_t hidden_units    = 75;

    using network_t = dll::dyn_network_desc<
        dll::network_layers<
            dll::rnn_layer<time_steps, sequence_length, hidden_units, dll::last_only, dll::truncate<20>>,
            dll::recurrent_last_layer<time_steps, hidden_units>,
            dll::dense_layer<hidden_units, 10, dll::softmax>
        >
        , dll::updater<dll::updater_type::ADAM>      // Adam
        , dll::batch_size<100>                       // The mini-batch size
    >::network_t;

    auto net = std::make_unique<network_t>();

    REQUIRE(net->fine_tune(dataset.train(), 30) < 0.15);
    REQUIRE(net->evaluate_error(dataset.test()) < 0.25);
}

// Deep RNN
DLL_TEST_CASE("unit/rnn/3", "[unit][rnn]") {
    auto dataset = dll::make_mnist_dataset_nc_sub(0, 2000, dll::batch_size<100>{}, dll::scale_pre<255>{});

    constexpr size_t time_steps      = 28;
    constexpr size_t sequence_length = 28;
    constexpr size_t hidden_units    = 30;

    using network_t = dll::dyn_network_desc<
        dll::network_layers<
            dll::rnn_layer<time_steps, sequence_length, hidden_units, dll::last_only>,
            dll::rnn_layer<time_steps, hidden_units, hidden_units, dll::last_only>,
            dll::recurrent_last_layer<time_steps, hidden_units>,
            dll::dense_layer<hidden_units, 10, dll::softmax>
        >
        , dll::updater<dll::updater_type::ADAM>      // Adam
        , dll::batch_size<100>                       // The mini-batch size
    >::network_t;

    auto net = std::make_unique<network_t>();

    REQUIRE(net->fine_tune(dataset.train(), 50) < 0.5);
    REQUIRE(net->evaluate_error(dataset.test()) < 0.5);
}
