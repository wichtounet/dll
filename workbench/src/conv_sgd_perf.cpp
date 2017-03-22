//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#define ETL_COUNTERS

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/dense_layer.hpp"
#include "dll/test.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(3000);

    auto n = dataset.training_images.size();
    std::cout << n << " samples to test" << std::endl;

    mnist::binarize_dataset(dataset);

    // Clean slate
    etl::reset_counters();
    dll::reset_timers();

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<1, 28, 28, 6, 5, 5>::layer_t,
            dll::conv_desc<6, 24, 24, 6, 5, 5>::layer_t,
            dll::dense_desc<6 * 20 * 20, 500>::layer_t,
            dll::dense_desc<500, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::momentum, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dbn_t>();

    // Train the network for performance sake
    net->display();
    net->fine_tune(dataset.training_images, dataset.training_labels, 20);

    std::cout << "DLL Timers" << std::endl;
    dll::dump_timers();

    std::cout << "ETL Counters" << std::endl;
    etl::dump_counters();

    return 0;
}
