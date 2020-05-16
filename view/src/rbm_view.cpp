//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "dll/rbm/rbm.hpp"
#include "dll/ocv_visualizer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    dll::rbm_desc<
        28 * 28, 10 * 10,
        dll::momentum,
        dll::trainer_rbm<dll::pcd1_trainer_t>,
        dll::batch_size<50>,
        dll::visible<dll::unit_type::GAUSSIAN>,
        dll::watcher<dll::opencv_rbm_visualizer>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();

    mnist::normalize_dataset(dataset);

    rbm.train(dataset.training_images, 500);

    return 0;
}
