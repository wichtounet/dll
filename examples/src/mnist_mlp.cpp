//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/dense_layer.hpp"
#include "dll/test.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>();

    // Limit the test
    dataset.training_images.resize(20000);
    dataset.training_labels.resize(20000);

    // Build the network

    using network_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 500>::layer_t,
            dll::dense_desc<500, 250>::layer_t,
            dll::dense_desc<250, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>
        , dll::momentum           // Use momentum during training
        , dll::batch_size<100>    // The mini-batch size
        , dll::normalize_pre
    >::dbn_t;

    auto net = std::make_unique<network_t>();

    // Display the network
    net->display();

    // Train the network for performance sake
    net->fine_tune(dataset.training_images, dataset.training_labels, 20);

    // Test the network on test set
    net->evaluate(dataset.test_images, dataset.test_labels);

    return 0;
}
