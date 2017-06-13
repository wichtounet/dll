//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/dense_layer.hpp"
#include "dll/test.hpp"
#include "dll/dbn.hpp"
#include "dll/datasets.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset(0, dll::batch_size<100>{}, dll::normalize_pre{});

    // Build the network

    using network_t = dll::dyn_dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 500>::layer_t,
            dll::dense_desc<500, 250>::layer_t,
            dll::dense_desc<250, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>
        , dll::updater<dll::updater_type::MOMENTUM>     // Momentum
        , dll::batch_size<100>                          // The mini-batch size
    >::dbn_t;

    auto net = std::make_unique<network_t>();

    // Display the network and dataset
    net->display();
    dataset.display();

    // Train the network for performance sake
    net->fine_tune(dataset.train(), 20);

    // Test the network on test set
    net->evaluate(dataset.test());

    return 0;
}
