//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#define DLL_SVM_SUPPORT

#include "dll/rbm/conv_rbm.hpp"
#include "dll/rbm/dyn_conv_rbm.hpp"
#include "dll/transform/rectifier_layer.hpp"
#include "dll/transform/lcn_layer.hpp"
#include "dll/transform/dyn_lcn_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/pooling/avgp_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("unit/cdbn/lcn/mnist/1", "[cdbn][lcn][svm][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::conv_rbm_square_desc<1, 28, 20, 17, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::conv_rbm_square_desc<20, 12, 20, 3, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::lcn_layer_desc<9>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(250);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 30);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

DLL_TEST_CASE("unit/cdbn/lcn/mnist/2", "[cdbn][lcn][svm][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::conv_rbm_square_desc<1, 28, 20, 17, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::rectifier_layer_desc<>::layer_t
            , dll::lcn_layer_desc<7>::layer_t
            , dll::conv_rbm_square_desc<20, 12, 20, 3, dll::visible<dll::unit_type::GAUSSIAN>, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->template layer_get<3>().learning_rate *= 3.0;
    dbn->template layer_get<3>().initial_momentum = 0.9;
    dbn->template layer_get<3>().momentum = 0.9;

    dbn->pretrain(dataset.training_images, 30);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.5); //Note: This is not very stable
}

DLL_TEST_CASE("unit/cdbn/lcn/mnist/3", "[cdbn][lcn][svm][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::conv_rbm_square_desc<1, 28, 20, 17, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::conv_rbm_square_desc<20, 12, 20, 3, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::rectifier_layer_desc<>::layer_t
            , dll::lcn_layer_desc<5>::layer_t
            , dll::mp_3d_layer_desc<20, 10, 10, 2, 2, 1>::layer_t
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

DLL_TEST_CASE("unit/cdbn/lcn/mnist/4", "[cdbn][lcn][svm][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::conv_rbm_square_desc<1, 28, 20, 17, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::conv_rbm_square_desc<20, 12, 20, 3, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::rectifier_layer_desc<>::layer_t
            , dll::lcn_layer_desc<5>::layer_t
            , dll::avgp_3d_layer_desc<20, 10, 10, 2, 2, 1>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error <= 0.12);
}

DLL_TEST_CASE("unit/cdbn/lcn/mnist/5", "[cdbn][lcn][svm][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::conv_rbm_square_desc<1, 28, 20, 17, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::conv_rbm_square_desc<20, 12, 20, 3, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::rectifier_layer_desc<>::layer_t
            , dll::lcn_layer_desc<7>::layer_t
            , dll::avgp_3d_layer_desc<20, 10, 10, 2, 2, 1>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(150);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<3>().sigma = 2.0;

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

DLL_TEST_CASE("unit/cdbn/lcn/mnist/6", "[cdbn][lcn][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::conv_rbm_square_desc<1, 28, 20, 17, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::lcn_layer_desc<5>::layer_t
            , dll::avgp_3d_layer_desc<20, 12, 12, 1, 2, 2>::layer_t
            , dll::conv_rbm_square_desc<20, 6, 20, 3, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::lcn_layer_desc<3>::layer_t
            , dll::avgp_3d_layer_desc<20, 4, 4, 1, 2, 2>::layer_t
        >>::dbn_t;

    REQUIRE(!dll::layer_traits<dbn_t::layer_type<1>>::is_pretrained());
    REQUIRE(!dll::layer_traits<dbn_t::layer_type<1>>::is_trained());
    REQUIRE(!dll::layer_traits<dbn_t::layer_type<2>>::is_pretrained());
    REQUIRE(!dll::layer_traits<dbn_t::layer_type<2>>::is_trained());

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(150);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<1>().sigma = 1.0;
    dbn->template layer_get<4>().sigma = 1.0;

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);
}

DLL_TEST_CASE("unit/cdbn/lcn/mnist/7", "[cdbn][lcn][svm][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::dyn_conv_rbm_desc<dll::momentum>::layer_t
            , dll::dyn_conv_rbm_desc<dll::momentum>::layer_t
            , dll::lcn_layer_desc<9>::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template init_layer<0>(1, 28, 28, 20, 17, 17);
    dbn->template init_layer<1>(20, 12, 12, 20, 3, 3);

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

DLL_TEST_CASE("unit/cdbn/lcn/mnist/8", "[cdbn][lcn][svm][unit]") {
    using dbn_t =
        dll::dbn_desc<dll::dbn_layers<
              dll::dyn_conv_rbm_desc<dll::momentum>::layer_t
            , dll::dyn_conv_rbm_desc<dll::momentum>::layer_t
            , dll::dyn_lcn_layer_desc::layer_t
        >>::dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template init_layer<0>(1, 28, 28, 20, 17, 17);
    dbn->template init_layer<1>(20, 12, 12, 20, 3, 3);
    dbn->template init_layer<2>(9);

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}
