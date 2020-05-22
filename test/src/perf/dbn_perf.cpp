//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/rbm/rbm.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dbn/perf/1", "[dbn][bench][fast]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<5>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<5>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<5>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<5>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(25);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 2);
    std::cout << "ft_error:" << ft_error << std::endl;

    auto test_error = dbn->evaluate_error(dataset.test_images, dataset.test_labels);
    std::cout << "test_error:" << test_error << std::endl;

    dll::dump_timers();
}

DLL_TEST_CASE("dbn/perf/3", "[dbn][bench][slow]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 300, dll::momentum, dll::batch_size<24>, dll::init_weights>::layer_t,
            dll::rbm_desc<300, 1000, dll::momentum, dll::batch_size<24>>::layer_t,
            dll::rbm_desc<1000, 10, dll::momentum, dll::batch_size<24>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<5>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(2000);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    dll::dump_timers();
}
