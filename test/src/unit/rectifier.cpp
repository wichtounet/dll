//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#define DLL_SVM_SUPPORT

#include "dll/rbm/dyn_conv_rbm.hpp"
#include "dll/rbm/conv_rbm.hpp"
#include "dll/transform/rectifier_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("unit/cdbn/rectifier/mnist/1", "[cdbn][rectifier][svm][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::conv_rbm_square_desc<1, 28, 20, 17, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::rectifier_layer_desc<>::layer_t
            , dll::conv_rbm_square_desc<20, 12, 20, 3, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

DLL_TEST_CASE("unit/cdbn/rectifier/mnist/2", "[cdbn][rectifier][svm][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::conv_rbm_square_desc<1, 28, 20, 17, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::rectifier_layer_desc<>::layer_t
            , dll::conv_rbm_square_desc<20, 12, 20, 3, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::rectifier_layer_desc<>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

DLL_TEST_CASE("unit/cdbn/rectifier/mnist/3", "[cdbn][rectifier][svm][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::dyn_conv_rbm_desc<dll::momentum>::layer_t
            , dll::rectifier_layer_desc<>::layer_t
            , dll::dyn_conv_rbm_desc<dll::momentum>::layer_t
            , dll::rectifier_layer_desc<>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(300);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(1, 28, 28, 10, 17, 17);
    dbn->template layer_get<2>().init_layer(10, 12, 12, 10, 3, 3);

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}
