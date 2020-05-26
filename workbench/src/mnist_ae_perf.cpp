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
            dll::dense_layer<28 * 28, 128, dll::relu>,
            dll::dense_layer<128, 28 * 28, dll::sigmoid>
        >
        , dll::batch_size<256>       // The mini-batch size
        , dll::shuffle               // Shuffle the dataset before each epoch
        , dll::binary_cross_entropy  // Use a Binary Cross Entropy Loss
        , dll::adadelta              // Adadelta updates for gradient descent
        , dll::no_batch_display      // Disable pretty print of each every batch
        , dll::no_epoch_error        // Disable computation of the error at each epoch
    >::network_t;

    auto net = std::make_unique<network_t>();

    // Display the network and dataset
    net->display_pretty();
    dataset.display_pretty();

    // Train the network as auto-encoder
    net->train_ae(dataset.train(), 10);

    // Test the network on test set
    net->evaluate_ae(dataset.test());

    // Show where the time was spent
    dll::dump_timers_pretty();

    // Show ETL performance counters
    etl::dump_counters_pretty();

    return 0;
}
