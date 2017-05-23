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

    // Scale the images

    for(auto& image : dataset.training_images){
        image /= 255.0;
    }

    for(auto& image : dataset.test_images){
        image /= 255.0;
    }

    // Build the network

    using network_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 32>::layer_t,
            dll::dense_desc<32, 28 * 28>::layer_t
        >,
        dll::momentum, dll::batch_size<100>, dll::loss<dll::loss_function::BINARY_CROSS_ENTROPY>>::dbn_t;

    auto net = std::make_unique<network_t>();

    // Display the network
    net->display();

    // Train the network as auto-encoder
    net->fine_tune_ae(dataset.training_images, 25);

    // Test the network on test set
    net->evaluate_ae(dataset.test_images);

    return 0;
}
