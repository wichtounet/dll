//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define ETL_COUNTERS
#define ETL_GPU_POOL

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_mnist_ae_dataset(dll::batch_size<256>{}, dll::scale_pre<255>{});

    // Build the network

    using network_t = dll::network_desc<
        dll::network_layers<
            dll::dense_layer_desc<784, 1024, dll::relu>::layer_t,
            dll::dense_layer_desc<1024, 256 , dll::relu>::layer_t,
            dll::dense_layer_desc<256 , 128 , dll::relu>::layer_t,
            // Encoded Features
            dll::dense_layer_desc<128 , 256 , dll::relu>::layer_t,
            dll::dense_layer_desc<256 , 1024, dll::relu>::layer_t,
            dll::dense_layer_desc<1024, 784, dll::sigmoid>::layer_t
        >
        , dll::batch_size<256>       // The mini-batch size
        , dll::shuffle               // Shuffle the dataset before each epoch
        , dll::binary_cross_entropy  // Use a Binary Cross Entropy Loss
        , dll::adadelta              // Adadelta updates for gradient descent
        , dll::no_batch_display      // Disable pretty print of each every batch
        , dll::no_epoch_error        // Disable computation of the error at each epoch
    >::network_t;

    auto net = std::make_unique<network_t>();

    // Display the network
    net->display();

    // Train the network as auto-encoder
    net->train_ae(dataset.train(), 5);

    // Test the network on test set
    net->evaluate_ae(dataset.test());

    // Show where the time was spent
    dll::dump_timers_pretty();

    // Show ETL performance counters
    etl::dump_counters_pretty();

    return 0;
}
