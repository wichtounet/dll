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

TEST_CASE( "unit/cdbn/mnist/1", "[cdbn][svm][unit]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<28, 1, 12, 20, dll::momentum, dll::batch_size<10>>::rbm_t,
        dll::conv_rbm_desc_square<12, 20, 10, 20, dll::momentum, dll::batch_size<10>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

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

TEST_CASE( "unit/cdbn/mnist/2", "[cdbn][svm][unit]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<28, 1, 12, 40, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<12, 40, 10, 40, dll::momentum, dll::batch_size<25>>::rbm_t>, dll::svm_concatenate>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);
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

TEST_CASE( "unit/cdbn/mnist/3", "[cdbn][gaussian][svm][unit]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<28, 1, 12, 40, dll::visible<dll::unit_type::GAUSSIAN>, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<12, 40, 10, 40, dll::momentum, dll::batch_size<25>>::rbm_t>, dll::svm_concatenate>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);
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

TEST_CASE( "unit/cdbn/mnist/4", "[cdbn][gaussian][svm][unit]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc_square<28, 1, 12, 40, dll::visible<dll::unit_type::GAUSSIAN>, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<12, 40, 10, 40, dll::momentum, dll::batch_size<25>>::rbm_t>,
        dll::svm_concatenate, dll::svm_scale>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);
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
