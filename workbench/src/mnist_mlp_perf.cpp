//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define ETL_COUNTERS
#define ETL_GPU_POOL

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/neural/dropout/dropout_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset(dll::batch_size<256>{}, dll::normalize_pre{});

    // Build the network

    using network_t = dll::network_desc<
        dll::network_layers<
            dll::dense_layer<28 * 28, 500>,
            dll::dropout_layer<50>,
            dll::dense_layer<500, 1000>,
            dll::dropout_layer<50>,
            dll::dense_layer<1000, 1000>,
            dll::dense_layer<1000, 10, dll::softmax>
        >
        , dll::updater<dll::updater_type::NADAM>     // Nesterov Adam (NADAM)
        , dll::batch_size<256>                       // The mini-batch size
        , dll::shuffle                               // Shuffle before each epoch
        , dll::no_batch_display                      // Disable pretty print of each every batch
        , dll::no_epoch_error                        // Disable computation of the error at each epoch
    >::network_t;

    auto net = std::make_unique<network_t>();

    // Display the network and dataset
    net->display_pretty();
    dataset.display_pretty();

    // Train the network for performance sake
    net->train(dataset.train(), 5);

    // Test the network on test set
    net->evaluate(dataset.test());

    // Show where the time was spent
    dll::dump_timers_pretty();

    // Show ETL performance counters
    etl::dump_counters_pretty();

    return 0;
}
