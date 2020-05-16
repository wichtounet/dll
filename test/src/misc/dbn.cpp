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
#include "dll/text_reader.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dbn/mnist_1", "dbn::simple") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<50>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10);
    std::cout << "ft_error:" << ft_error << std::endl;
    REQUIRE(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("dbn/mnist_2", "dbn::containers") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<50>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(200);
    dataset.training_labels.resize(200);

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 5);

    REQUIRE(error < 5e-2);
}

DLL_TEST_CASE("dbn/mnist_3", "dbn::labels") {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1000);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    typedef dll::dbn_desc<
        dll::dbn_label_layers<
            dll::rbm_desc<28 * 28, 200, dll::batch_size<50>, dll::init_weights, dll::momentum>::layer_t,
            dll::rbm_desc<200, 300, dll::batch_size<50>, dll::momentum>::layer_t,
            dll::rbm_desc<310, 500, dll::batch_size<50>, dll::momentum>::layer_t>,
        dll::batch_size<10>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 10);

    auto error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::label_predictor());
    std::cout << "test_error:" << error << std::endl;
    REQUIRE(error < 0.3);
}

DLL_TEST_CASE("dbn/mnist_6", "dbn::cg_gaussian") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<25>, dll::visible<dll::unit_type::GAUSSIAN>>::layer_t,
            dll::rbm_desc<200, 500, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<500, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<50>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1000);

    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10);

    REQUIRE(error < 5e-2);

    TEST_CHECK(0.2);
}

//This test should not perform well, but should not fail
DLL_TEST_CASE("dbn/mnist_8", "dbn::cg_relu") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::RELU>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<50>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(200);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10);

    REQUIRE(std::isfinite(error));

    auto test_error = dbn->evaluate_error(dataset.test_images, dataset.test_labels);
    std::cout << "test_error:" << test_error << std::endl;
}

DLL_TEST_CASE("dbn/mnist_17", "dbn::memory") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_mode, dll::trainer<dll::cg_trainer>, dll::batch_size<50>, dll::big_batch_size<3>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(1078);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto error = dbn->fine_tune(
        dataset.training_images.begin(), dataset.training_images.end(),
        dataset.training_labels.begin(), dataset.training_labels.end(),
        10);

    REQUIRE(error < 5e-2);

    TEST_CHECK(0.2);

    //Mostly here to ensure compilation
    auto out = dbn->prepare_one_output<etl::dyn_matrix<float, 1>>();
    REQUIRE(out.size() > 0);
}

DLL_TEST_CASE("dbn/mnist/text/1", "dbn::simple") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<50>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto training_images = dll::text::read_images<std::vector, etl::dyn_matrix<float, 1>, false>(
        "/home/wichtounet/datasets/mnist_text/train/images", 500);

    auto test_images = dll::text::read_images<std::vector, etl::dyn_matrix<float, 1>, false>(
        "/home/wichtounet/datasets/mnist_text/test/images", 500);

    auto training_labels = dll::text::read_labels<std::vector, uint8_t>("/home/wichtounet/datasets/mnist_text/train/labels", 500);
    auto test_labels = dll::text::read_labels<std::vector, uint8_t>("/home/wichtounet/datasets/mnist_text/test/labels", 500);

    REQUIRE(training_images.size() == 500);
    REQUIRE(test_images.size() == 500);
    REQUIRE(training_labels.size() == 500);
    REQUIRE(test_labels.size() == 500);

    mnist::binarize_each(training_images);
    mnist::binarize_each(test_images);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(training_images, 20);

    auto error = dbn->fine_tune(training_images, training_labels, 10);
    REQUIRE(error < 5e-2);

    auto test_error = dbn->evaluate_error(test_images, test_labels);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

DLL_TEST_CASE("mnist_original", "dbn::simple") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 500, dll::momentum, dll::batch_size<64>, dll::init_weights>::layer_t,
            dll::rbm_desc<500, 500, dll::momentum, dll::batch_size<64>>::layer_t,
            dll::rbm_desc<500, 2000, dll::momentum, dll::batch_size<64>>::layer_t,
            dll::rbm_desc<2000, 10, dll::momentum, dll::batch_size<64>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<300>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>();

    REQUIRE(!dataset.training_images.empty());
    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 10);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 10);
    std::cout << "ft_error:" << ft_error << std::endl;
    REQUIRE(ft_error < 5e-2);

    TEST_CHECK(0.2);
}
