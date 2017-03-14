//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#define DLL_SVM_SUPPORT

#include "dll/augment/augment_layer.hpp"
#include "dll/transform/rectifier_layer.hpp"
#include "dll/transform/random_layer.hpp"
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

TEST_CASE("unit/cdbn/mnist/7", "[cdbn][ap][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<1, 28, 28, 20, 9, 8, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::avgp_layer_3d_desc<20, 20, 21, 2, 2, 3>::layer_t,
            dll::conv_rbm_desc<10, 10, 7, 20, 3, 3, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::avgp_layer_3d_desc<20, 8, 5, 2, 1, 1>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    REQUIRE(dbn->output_size() == 400);

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
            dll::conv_rbm_desc<1, 28, 28, 20, 15, 17, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_desc<20, 14, 12, 20, 7, 3, dll::momentum, dll::batch_size<25>>::layer_t, dll::mp_layer_3d_desc<20, 8, 10, 1, 1, 1>::layer_t, dll::avgp_layer_3d_desc<20, 8, 10, 1, 1, 1>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    REQUIRE(dbn->output_size() == 1600);

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
        dll::batch_mode>::dbn_t dbn_t;

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

TEST_CASE("unit/cdbn/mnist/12", "[dbn][conv][mnist][patches][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_patches_layer_padh_desc<>::layer_t,
            dll::dyn_conv_rbm_desc<dll::momentum>::layer_t,
            dll::dyn_conv_rbm_desc<dll::momentum>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(50);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template init_layer<0>(14, 14, 14, 14, 1);
    dbn->template init_layer<1>(1, 14, 14, 20, 5, 5);
    dbn->template init_layer<2>(20, 10, 10, 20, 5, 5);

    dbn->pretrain(dataset.training_images, 10);

    auto probs = dbn->activation_probabilities(dataset.training_images[0]);
    REQUIRE(probs.size() == 4);
}

TEST_CASE("hybrid/mnist/9", "[cdbn][augment][unit]") {
    using dbn_t =
        dll::dyn_dbn_desc<dll::dbn_layers<
            dll::augment_layer_desc<dll::copy<2>, dll::copy<3>>::layer_t,
            dll::conv_rbm_desc_square<1, 28, 20, 8, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    REQUIRE(dbn->activation_probabilities(dataset.training_images[0]).size() > 0);
}

TEST_CASE("hybrid/mnist/5", "[cdbn][rectifier][svm][unit]") {
    using dbn_t =
        dll::dyn_dbn_desc<dll::dbn_layers<
              dll::conv_rbm_desc_square<1, 28, 20, 12, dll::parallel_mode, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::random_layer_desc::layer_t
            , dll::rectifier_layer_desc<>::layer_t
            , dll::conv_rbm_desc_square<20, 12, 20, 10, dll::parallel_mode, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();
    dbn->display();
}
