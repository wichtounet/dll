//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/neural/rnn/rnn_layer.hpp"
#include "dll/neural/recurrent/recurrent_last_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset_nc(dll::batch_size<100>{}, dll::scale_pre<255>{});

    constexpr size_t time_steps      = 28;
    constexpr size_t sequence_length = 28;
    constexpr size_t hidden_units    = 100;

    // Build the network

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

    // Display the network and dataset
    net->display();
    dataset.display();

    // Train the network for performance sake
    net->train(dataset.train(), 50);

    // Test the network on test set
    net->evaluate(dataset.test());

    return 0;
}
