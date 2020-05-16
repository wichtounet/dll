//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#define DLL_SVM_SUPPORT

#include "dll/transform/rectifier_layer.hpp"
#include "dll/transform/random_layer.hpp"
#include "dll/rbm/conv_rbm.hpp"
#include "dll/rbm/dyn_conv_rbm.hpp"
#include "dll/rbm/conv_rbm_mp.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/pooling/avgp_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("unit/cdbn/mnist/7", "[cdbn][ap][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<1, 28, 28, 16, 9, 9, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::avgp_3d_layer_desc<16, 20, 20, 1, 2, 2>::layer_t,
            dll::conv_rbm_desc<16, 10, 10, 8, 3, 3, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::avgp_3d_layer_desc<8, 8, 8, 1, 2, 2>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    REQUIRE(dbn->output_size() == 8 * 4 * 4);

    dbn->pretrain(dataset.training_images, 25);

    auto output = dbn->forward_one(dataset.training_images.front());
    REQUIRE(output.size() == 8 * 4 * 4);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error <= 0.6);
}

DLL_TEST_CASE("unit/cdbn/mnist/8", "[cdbn][ap][svm][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_desc<1, 28, 28, 20, 15, 17, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_desc<20, 14, 12, 20, 7, 3, dll::momentum, dll::batch_size<25>>::layer_t, dll::mp_3d_layer_desc<20, 8, 10, 1, 1, 1>::layer_t, dll::avgp_3d_layer_desc<20, 8, 10, 1, 1, 1>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    REQUIRE(dbn->output_size() == 1600);

    dbn->pretrain(dataset.training_images, 20);

    auto output = dbn->forward_one(dataset.training_images.front());
    REQUIRE(output.size() == 1600);

    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);
    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}

DLL_TEST_CASE("hybrid/mnist/5", "[cdbn][rectifier][svm][unit]") {
    using dbn_t =
        dll::dyn_dbn_desc<dll::dbn_layers<
              dll::conv_rbm_square_desc<1, 28, 20, 17, dll::momentum, dll::batch_size<10>>::layer_t
            , dll::random_layer_desc::layer_t
            , dll::rectifier_layer_desc<>::layer_t
            , dll::conv_rbm_square_desc<20, 12, 20, 3, dll::momentum, dll::batch_size<10>>::layer_t
        >>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();
    dbn->display();
}
