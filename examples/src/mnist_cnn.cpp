//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/conv_same_layer.hpp"
#include "dll/neural/dense_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>();

    // Limit the test
    dataset.training_images.resize(20000);
    dataset.training_labels.resize(20000);

    mnist::binarize_dataset(dataset);

    // Build the network

    using network_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_same_desc<1, 28, 28, 6, 3, 3>::layer_t,
            dll::mp_layer_3d_desc<6, 28, 28, 1, 2, 2>::layer_t,
            dll::conv_same_desc<6, 14, 14, 6, 3, 3>::layer_t,
            dll::mp_layer_3d_desc<6, 14, 14, 1, 2, 2>::layer_t,
            dll::dense_desc<6 * 7 * 7, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::momentum, dll::batch_size<100>, dll::shuffle>::dbn_t;

    auto net = std::make_unique<network_t>();

    net->learning_rate = 0.05;

    // Display the network
    net->display();

    // Train the network for performance sake
    net->fine_tune(dataset.training_images, dataset.training_labels, 25);

    // Test the network on test set
    net->evaluate(dataset.test_images, dataset.test_labels);

    return 0;
}
