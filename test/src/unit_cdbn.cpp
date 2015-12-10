//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#define DLL_SVM_SUPPORT

#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"
#include "dll/mp_layer.hpp"
#include "dll/avgp_layer.hpp"
#include "dll/patches_layer.hpp"
#include "dll/patches_layer_pad.hpp"

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
            dll::conv_rbm_desc_square<1, 28, 20, 12, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_desc_square<20, 12, 20, 10, dll::momentum, dll::batch_size<25>>::layer_t>,
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
    REQUIRE(test_error < 0.1);
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
            dll::conv_rbm_desc<1, 28, 28, 20, 20, 21, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::mp_layer_3d_desc<20, 20, 21, 2, 2, 3>::layer_t,
            dll::conv_rbm_desc<10, 10, 7, 20, 8, 5, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::mp_layer_3d_desc<20, 8, 5, 2, 1, 1>::layer_t>>::dbn_t dbn_t;

    REQUIRE(dbn_t::output_size() == 400);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(250);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto output = dbn->activation_probabilities(dataset.training_images.front());
    REQUIRE(output.size() == 400);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.5);
}

TEST_CASE("unit/cdbn/mnist/7", "[cdbn][ap][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<1, 28, 28, 20, 20, 21, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::avgp_layer_3d_desc<20, 20, 21, 2, 2, 3>::layer_t,
            dll::conv_rbm_desc<10, 10, 7, 20, 8, 5, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::avgp_layer_3d_desc<20, 8, 5, 2, 1, 1>::layer_t>>::dbn_t dbn_t;

    REQUIRE(dbn_t::output_size() == 400);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto output = dbn->activation_probabilities(dataset.training_images.front());
    REQUIRE(output.size() == 400);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.5);
}

TEST_CASE("unit/cdbn/mnist/8", "[cdbn][ap][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<1, 28, 28, 20, 14, 12, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_desc<20, 14, 12, 20, 8, 10, dll::momentum, dll::batch_size<25>>::layer_t, dll::mp_layer_3d_desc<20, 8, 10, 1, 1, 1>::layer_t, dll::avgp_layer_3d_desc<20, 8, 10, 1, 1, 1>::layer_t>>::dbn_t dbn_t;

    REQUIRE(dbn_t::output_size() == 1600);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto output = dbn->activation_probabilities(dataset.training_images.front());
    REQUIRE(output.size() == 1600);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

TEST_CASE("unit/cdbn/mnist/9", "[dbn][conv][mnist][patches][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::patches_layer_desc<14, 14, 14, 14>::layer_t,
            dll::conv_rbm_desc_square<1, 14, 10, 10, dll::parallel_mode, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::conv_rbm_desc_square<10, 10, 10, 6, dll::parallel_mode, dll::momentum, dll::batch_size<10>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(50);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 10);

    auto probs = dbn->activation_probabilities(dataset.training_images[0]);
    REQUIRE(probs.size() == 4);
}

TEST_CASE("unit/cdbn/mnist/10", "[dbn][conv][mnist][patches][memory][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::patches_layer_desc<14, 14, 14, 14>::layer_t,
            dll::conv_rbm_desc_square<1, 14, 20, 10, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::conv_rbm_desc_square<20, 10, 20, 6, dll::momentum, dll::batch_size<10>>::layer_t>,
        dll::memory>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(50);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 10);

    auto probs = dbn->activation_probabilities(dataset.training_images[0]);
    REQUIRE(probs.size() == 4);

    //Simply to ensure compilation
    if (false) {
        dbn->display();
        dbn->store("test.dat");
        dbn->load("test.dat");
    }
}

TEST_CASE("unit/cdbn/mnist/11", "[dbn][conv][mnist][patches][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::patches_layer_padh_desc<14, 14, 14, 14, 1>::layer_t,
            dll::conv_rbm_desc_square<1, 14, 20, 10, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::conv_rbm_desc_square<20, 10, 20, 6, dll::momentum, dll::batch_size<10>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(50);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 10);

    auto probs = dbn->activation_probabilities(dataset.training_images[0]);
    REQUIRE(probs.size() == 4);
}
