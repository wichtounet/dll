//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#define DLL_SVM_SUPPORT

#include "dll/conv_rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/mp_layer.hpp"
#include "dll/avgp_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "conv_dbn/mnist_1", "conv_dbn::simple" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<1, 28, 40, 12, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<40, 12, 20, 10, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<20, 10, 50, 6, dll::momentum, dll::batch_size<25>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
}

TEST_CASE( "conv_dbn/mnist_2", "conv_dbn::svm_simple" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<1, 28, 40, 12, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<40, 12, 40, 10, dll::momentum, dll::batch_size<25>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE( "conv_dbn/mnist_3", "conv_dbn::svm_concatenate" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<1, 28, 40, 12, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<40, 12, 40, 10, dll::momentum, dll::batch_size<25>>::rbm_t>, dll::svm_concatenate>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE( "conv_dbn/mnist_4", "conv_dbn::svm_simple" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<1, 28, 40, 12, dll::momentum, dll::batch_size<25>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE( "conv_dbn/mnist_5", "conv_dbn::svm_simple" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<1, 28, 40, 12, dll::momentum, dll::batch_size<25>>::rbm_t>, dll::svm_concatenate>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE( "conv_dbn/mnist_6", "conv_dbn::svm_gaussian" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<1, 28, 20, 20, dll::visible<dll::unit_type::GAUSSIAN>, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<20, 20, 20, 16, dll::momentum, dll::batch_size<25>>::rbm_t>, dll::svm_concatenate>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE( "conv_dbn/mnist_7", "conv_dbn::svm_scale" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<1, 28, 40, 12, dll::visible<dll::unit_type::GAUSSIAN>, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<40, 12, 40, 10, dll::momentum, dll::batch_size<25>>::rbm_t>,
        dll::svm_concatenate, dll::svm_scale>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(333);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE( "conv_dbn/mnist_8", "conv_dbn::unsquare_svm" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc<1, 28, 28, 40, 14, 12, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc<40, 14, 12, 40, 8, 10, dll::momentum, dll::batch_size<25>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE( "conv_dbn/mnist_slow", "[cdbn][slow][benchmark]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<1, 28, 40, 12, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<40, 12, 40, 10, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<40, 10, 40, 6, dll::momentum, dll::batch_size<25>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(1000);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
}
