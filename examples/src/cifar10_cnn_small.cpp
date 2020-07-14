//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define ETL_COUNTERS
#define ETL_GPU_TOOLS
#define ETL_GPU_POOL

#include "dll/neural/conv/conv_layer.hpp"
#include "dll/neural/dense/dense_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_cifar10_dataset(dll::batch_size<256>{}, dll::scale_pre<255>{});

    using dbn_t = dll::dbn_desc<
            dll::dbn_layers<
                    dll::conv_layer<3, 32, 32, 12, 5, 5, dll::relu>,
                    dll::mp_3d_layer<12, 28, 28, 1, 2, 2>,
                    dll::conv_layer<12, 14, 14, 24, 3, 3, dll::relu>,
                    dll::mp_3d_layer<24, 12, 12, 1, 2, 2>,
                    dll::dense_layer<24 * 6 * 6, 64, dll::relu>,
                    dll::dense_layer<64, 10, dll::softmax>
            >,
            dll::updater<dll::updater_type::MOMENTUM>,
            dll::batch_size<256>,
            dll::no_batch_display,
            dll::no_epoch_error
    >::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.001;
    dbn->initial_momentum = 0.9;
    dbn->momentum = 0.9;
    dbn->goal = -1.0;

    dbn->display();

    dbn->fine_tune(dataset.train(), 5);

    dbn->evaluate(dataset.test());

    dll::dump_timers_pretty();

    etl::dump_counters_pretty();

    return 0;
}