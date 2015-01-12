//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "catch.hpp"

#define DLL_SVM_SUPPORT

#include "dll/conv_rbm_mp.hpp"
#include "dll/conv_dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "conv_dbn_mp/mnist_1", "conv_dbn::simple" ) {
    typedef dll::conv_dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_mp_desc<28, 1, 12, 40, 2, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_mp_desc<6, 40, 4, 20, 2, dll::momentum, dll::batch_size<25>>::rbm_t
    >>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
}

TEST_CASE( "conv_dbn_mp/mnist_2", "conv_dbn::svm_simple" ) {
    typedef dll::conv_dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_mp_desc<28, 1, 18, 40, 2, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_mp_desc<9, 40, 6, 40, 2, dll::momentum, dll::batch_size<25>>::rbm_t
    >>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE( "conv_dbn_mp/mnist_3", "conv_dbn::svm_concatenate" ) {
    typedef dll::conv_dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_mp_desc<28, 1, 18, 40, 2, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_mp_desc<9, 40, 6, 40, 2, dll::momentum, dll::batch_size<25>>::rbm_t
    >, dll::concatenate>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}
