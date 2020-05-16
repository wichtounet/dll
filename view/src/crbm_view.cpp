//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "dll/rbm/conv_rbm.hpp"
#include "dll/ocv_visualizer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    dll::conv_rbm_square_desc<
        1, 28, 40, 17,
        dll::momentum,
        dll::batch_size<50>,
        dll::sparsity<dll::sparsity_method::LEE>,
        dll::watcher<dll::opencv_rbm_visualizer>>::layer_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    dataset.training_images.resize(500);
    dataset.training_labels.resize(500);

    mnist::binarize_dataset(dataset);

    rbm.train(dataset.training_images, 500);

    return 0;
}
