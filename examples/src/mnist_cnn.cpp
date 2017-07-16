//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/dense_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/datasets.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset(0, dll::batch_size<100>{}, dll::scale_pre<255>{});

    // Build the network

    using network_t = dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 8, 5, 5>::layer_t,
            dll::mp_layer_2d_desc<8, 24, 24, 2, 2>::layer_t,
            dll::conv_desc<8, 12, 12, 8, 5, 5>::layer_t,
            dll::mp_layer_2d_desc<8, 8, 8, 2, 2>::layer_t,
            dll::dense_desc<8 * 4 * 4, 150>::layer_t,
            dll::dense_desc<150, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>
        , dll::updater<dll::updater_type::MOMENTUM>     // Momentum
        , dll::batch_size<100>                          // The mini-batch size
        , dll::shuffle                                  // Shuffle the dataset before each epoch
    >::dbn_t;

    auto net = std::make_unique<network_t>();

    net->learning_rate = 0.1;

    // Display the network and dataset
    net->display();
    dataset.display();

    // Train the network
    net->fine_tune(dataset.train(), 25);

    // Test the network on test set
    net->evaluate(dataset.test());

    return 0;
}
