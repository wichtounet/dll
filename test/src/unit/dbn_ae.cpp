//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#define DLL_SVM_SUPPORT

#include "dll/rbm/rbm.hpp"
#include "dll/rbm/dyn_rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/transform/shape_1d_layer.hpp"
#include "dll/transform/binarize_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dbn/ae/1", "[unit][rbm][dbn][mnist][sgd][ae]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 32, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<32, 28 * 28, dll::momentum, dll::batch_size<25>>::layer_t
        >, dll::autoencoder, dll::loss<dll::loss_function::BINARY_CROSS_ENTROPY>, dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->display();

    dbn->pretrain(dataset.training_images, 20);

    dbn->learning_rate = 0.1;

    auto ft_error = dbn->fine_tune_ae(dataset.training_images, 25);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 0.1);

    auto test_error = dll::test_set_ae(*dbn, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.1);
}
