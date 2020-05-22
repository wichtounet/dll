//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#define ETL_COUNTERS

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/test.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // First experiment : Dense - Dense - Dense
    // Current speed on frigg:
    //   ~20 seconds (mkl, default options)
    //   ~13 seconds (mkl-threads, default options)

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();
    dataset.training_images.resize(10000);
    dataset.training_labels.resize(10000);

    auto n = dataset.training_images.size();
    std::cout << n << " samples to test" << std::endl;

    mnist::binarize_dataset(dataset);

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 500>::layer_t,
            dll::dense_layer_desc<500, 250>::layer_t,
            dll::dense_layer_desc<250, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

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
