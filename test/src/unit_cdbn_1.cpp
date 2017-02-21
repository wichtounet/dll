//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#define DLL_SVM_SUPPORT

#include "dll/rbm/conv_rbm.hpp"
#include "dll/rbm/dyn_conv_rbm.hpp"
#include "dll/rbm/conv_rbm_mp.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/pooling/avgp_layer.hpp"
#include "dll/patches/patches_layer.hpp"
#include "dll/patches/patches_layer_pad.hpp"
#include "dll/patches/dyn_patches_layer_pad.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/cdbn/mnist/1", "[cdbn][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc_square<1, 28, 20, 12, dll::parallel_mode, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::conv_rbm_desc_square<20, 12, 20, 10, dll::parallel_mode, dll::momentum, dll::batch_size<10>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
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

TEST_CASE("unit/cdbn/mnist/2", "[cdbn][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc_square<1, 28, 10, 12, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_desc_square<10, 12, 10, 10, dll::momentum, dll::batch_size<25>>::layer_t>,
        dll::svm_concatenate>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);
    svm::rbf_grid g;
    g.c_steps     = 5;
    g.gamma_steps = 5;

    auto gs_result = dbn->svm_grid_search(dataset.training_images, dataset.training_labels, 3, g);
    REQUIRE(gs_result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE("unit/cdbn/mnist/3", "[cdbn][gaussian][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc_square<1, 28, 20, 12, dll::visible<dll::unit_type::GAUSSIAN>, dll::momentum, dll::batch_size<20>>::layer_t,
            dll::conv_rbm_desc_square<20, 12, 20, 10, dll::momentum, dll::batch_size<20>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 25);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE("unit/cdbn/mnist/4", "[cdbn][gaussian][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc_square<1, 28, 20, 12, dll::visible<dll::unit_type::GAUSSIAN>, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_desc_square<20, 12, 20, 10, dll::momentum, dll::batch_size<25>>::layer_t>,
        dll::svm_scale>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

TEST_CASE("unit/cdbn/mnist/5", "[cdbn][crbm_mp][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_mp_desc_square<1, 28, 20, 18, 2, dll::momentum, dll::batch_size<8>>::layer_t,
            dll::conv_rbm_mp_desc_square<20, 9, 20, 6, 2, dll::momentum, dll::batch_size<8>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
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

TEST_CASE("unit/cdbn/mnist/6", "[cdbn][mp][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<1, 28, 28, 10, 20, 21, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::mp_layer_3d_desc<10, 20, 21, 2, 2, 3>::layer_t,
            dll::conv_rbm_desc<5, 10, 7, 10, 8, 5, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::mp_layer_3d_desc<10, 8, 5, 2, 1, 1>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(250);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    REQUIRE(dbn->output_size() == 200);

    dbn->pretrain(dataset.training_images, 20);

    auto output = dbn->activation_probabilities(dataset.training_images.front());
    REQUIRE(output.size() == 200);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.5);
}
