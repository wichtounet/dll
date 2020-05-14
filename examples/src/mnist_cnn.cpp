//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/dense_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset(dll::batch_size<100>{}, dll::scale_pre<255>{});

    // Build the network

    using network_t = dll::dyn_network_desc<
        dll::network_layers<
            dll::conv_layer<1, 28, 28, 8, 5, 5>,
            dll::mp_2d_layer<8, 24, 24, 2, 2>,
            dll::conv_layer<8, 12, 12, 8, 5, 5>,
            dll::mp_2d_layer<8, 8, 8, 2, 2>,
            dll::dense_layer<8 * 4 * 4, 150>,
            dll::dense_layer<150, 10, dll::softmax>
        >
        , dll::updater<dll::updater_type::NADAM>     // Momentum
        , dll::batch_size<100>                       // The mini-batch size
        , dll::shuffle                               // Shuffle the dataset before each epoch
    >::network_t;

    auto net = std::make_unique<network_t>();

    // Display the network and dataset
    net->display();
    dataset.display();

    // Train the network
    net->train(dataset.train(), 25);

    // Test the network on test set
    net->evaluate(dataset.test());

    return 0;
}
