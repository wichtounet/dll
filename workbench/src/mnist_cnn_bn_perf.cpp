//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define ETL_COUNTERS
#define ETL_GPU_POOL

#include "dll/neural/conv/conv_layer.hpp"
#include "dll/neural/dense/dense_layer.hpp"
#include "dll/neural/bn/batch_normalization_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset(dll::batch_size<256>{}, dll::scale_pre<255>{});

    // Build the network

    using network_t = dll::network_desc<
        dll::network_layers<
            dll::conv_layer<1, 28, 28, 16, 5, 5>,
            dll::mp_2d_layer<16, 24, 24, 2, 2>,
            dll::batch_normalization_4d_layer_desc<16, 12, 12>::layer_t,

            dll::conv_layer<16, 12, 12, 16, 5, 5>,
            dll::mp_2d_layer<16, 8, 8, 2, 2>,
            dll::batch_normalization_4d_layer_desc<16, 4, 4>::layer_t,

            dll::dense_layer<16 * 4 * 4, 256>,
            dll::batch_normalization_2d_layer_desc<256>::layer_t,

            dll::dense_layer<256, 10, dll::softmax>
        >
        , dll::updater<dll::updater_type::ADADELTA>  // ADADELTA
        , dll::batch_size<256>                       // The mini-batch size
        , dll::shuffle                               // Shuffle the dataset before each epoch
        , dll::no_batch_display                    // Disable pretty print of each every batch
        , dll::no_epoch_error                      // Disable computation of the error at each epoch
        , dll::early_training                        // Do not use validation error for early stopping (BN)
    >::network_t;

    auto net = std::make_unique<network_t>();

    // Display the network and dataset
    net->display_pretty();
    dataset.display_pretty();

    // Train the network
    net->train(dataset.train(), 10);

    // Test the network on test set
    net->evaluate(dataset.test());

    // Show where the time was spent
    dll::dump_timers_pretty();

    // Show ETL performance counters
    etl::dump_counters_pretty();

    return 0;
}
