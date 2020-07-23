//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define ETL_COUNTERS
#define ETL_GPU_POOL

#include <dll/neural/conv/conv_same_layer.hpp>

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_cifar10_dataset(dll::batch_size<256>{}, dll::scale_pre<255>{});

    using network_t = dll::network_desc<
        dll::network_layers<
            dll::conv_same_layer<3, 32, 32, 12, 5, 5, dll::relu>,
            dll::conv_same_layer<12, 32, 32, 12, 3, 3, dll::relu>,
            dll::mp_3d_layer<12, 32, 32, 1, 2, 2>,
            dll::conv_same_layer<12, 16, 16, 24, 5, 5, dll::relu>,
            dll::conv_same_layer<24, 16, 16, 24, 3, 3, dll::relu>,
            dll::mp_3d_layer<24, 16, 16, 1, 2, 2>,
            dll::conv_same_layer<24, 8, 8, 48, 3, 3, dll::relu>,
            dll::conv_same_layer<48, 8, 8, 48, 3, 3, dll::relu>,
            dll::mp_3d_layer<48, 8, 8, 1, 2, 2>,
            dll::dense_layer<48 * 4 * 4, 64, dll::relu>,
            dll::dense_layer<64, 10, dll::softmax>
        >,
        dll::updater<dll::updater_type::MOMENTUM>,
        dll::batch_size<256>,
        dll::no_batch_display,
        dll::no_epoch_error
    >::network_t;

    auto net = std::make_unique<network_t>();

    // Display the network and dataset
    net->display_pretty();
    dataset.display_pretty();

    // Train the network
    net->train(dataset.train(), 5);

    // Test the network on test set
    net->evaluate(dataset.test());

    // Show where the time was spent
    dll::dump_timers_pretty();

    // Show ETL performance counters
    etl::dump_counters_pretty();

    return 0;
}
