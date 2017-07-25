//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/dense_layer.hpp"
#include "dll/test.hpp"
#include "dll/network.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>();

    // Build the network

    using network_t = dll::dyn_network_desc<
        dll::network_layers<
            dll::dense_desc<28 * 28, 32, dll::relu>::layer_t,
            dll::dense_desc<32, 28 * 28, dll::sigmoid>::layer_t
        >
        , dll::batch_size<256>       // The mini-batch size
        , dll::shuffle               // Shuffle the dataset before each epoch
        , dll::binary_cross_entropy  // Use a Binary Cross Entropy Loss
        , dll::scale_pre<255>        // Scale the images (divide by 255)
        , dll::autoencoder           // Indicate auto-encoder
        , dll::adadelta              // Adadelta updates for gradient descent
    >::network_t;

    auto net = std::make_unique<network_t>();

    // Display the network
    net->display();

    // Train the network as auto-encoder
    net->fine_tune_ae(dataset.training_images, 50);

    // Test the network on test set
    net->evaluate_ae(dataset.test_images);

    return 0;
}
