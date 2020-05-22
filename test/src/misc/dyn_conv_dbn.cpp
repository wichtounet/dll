//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#define DLL_SVM_SUPPORT

#include "dll/rbm/dyn_conv_rbm.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dyn_conv_dbn/mnist_1", "conv_dbn::simple") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dyn_conv_rbm_desc<dll::momentum, dll::batch_size<25>>::layer_t,
            dll::dyn_conv_rbm_desc<dll::momentum, dll::batch_size<25>>::layer_t,
            dll::dyn_conv_rbm_desc<dll::momentum, dll::batch_size<25>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(1, 28, 28, 40, 17, 17);
    dbn->template layer_get<1>().init_layer(40, 12, 12, 20, 3, 3);
    dbn->template layer_get<2>().init_layer(20, 10, 10, 50, 5, 5);

    dbn->pretrain(dataset.training_images, 5);
}
