//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// This test case is done to benchmark the performance of SGD
DLL_TEST_CASE("dbn/sgd/perf/1", "[dbn][mnist][sgd][perf]") {
    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 500>::layer_t,
            dll::dense_layer_desc<500, 250>::layer_t,
            dll::dense_layer_desc<250, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(2000);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;
    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}
