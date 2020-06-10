//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/rbm/conv_rbm.hpp"
#include "dll/rbm/rbm.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto ae_dataset = dll::make_mnist_ae_dataset(dll::batch_size<100>{}, dll::binarize_pre<30>{});
    auto dataset = dll::make_mnist_dataset(dll::batch_size<100>{}, dll::binarize_pre<30>{});

    // Build the network

    using network_t = dll::network_desc<
        dll::network_layers<
            dll::conv_rbm_square_desc<1, 28, 8, 9, dll::momentum, dll::batch_size<100>>::layer_t,
            dll::conv_rbm_square_desc<8, 20, 8, 5, dll::momentum, dll::batch_size<100>>::layer_t,
            dll::rbm<8 * 16 * 16, 1000, dll::batch_size<100>, dll::momentum>,
            dll::rbm<1000, 10, dll::batch_size<100>, dll::hidden<dll::unit_type::SOFTMAX>>
        >
        , dll::updater<dll::updater_type::NADAM>     // Nesterov Adam (NADAM)
        , dll::batch_size<100>                       // The mini-batch size
        , dll::shuffle                               // Shuffle before each epoch
    >::network_t;

    auto net = std::make_unique<network_t>();

    // Display the network and dataset
    net->display_pretty();
    dataset.display_pretty();

    // Pretrain the network with contrastive divergence
    net->pretrain(ae_dataset.train(), 10);

    // Train the network for performance sake
    net->fine_tune(dataset.train(), 50);

    // Test the network on test set
    net->evaluate(dataset.test());

    // Show where the time was spent
    dll::dump_timers_pretty();

    return 0;
}
