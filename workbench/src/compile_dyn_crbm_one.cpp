//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/rbm/conv_rbm.hpp"
#include "dll/rbm/rbm.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// 1 6-layer networks

int main(int, char**) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(20000);

    mnist::binarize_dataset(dataset);

#define decl_dbn6(NAME, NAME_T, F)                                                                \
    using NAME_T =                                                                                \
        dll::dbn_desc<                                                                            \
            dll::dbn_layers<                                                                      \
                dll::dyn_conv_rbm_desc<dll::momentum>::layer_t,                                   \
                dll::dyn_mp_3d_layer_desc<>::layer_t,                                             \
                dll::dyn_conv_rbm_desc<dll::momentum>::layer_t,                                   \
                dll::dyn_mp_3d_layer_desc<>::layer_t,                                             \
                dll::dyn_rbm_desc<dll::momentum>::layer_t,                                        \
                dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>, \
            dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<64>>::dbn_t;           \
    auto NAME = std::make_unique<NAME_T>();                                                       \
    NAME->template layer_get<0>().init_layer(1, 28, 28, 10 + F, 5, 5);                            \
    NAME->template layer_get<1>().init_layer(10 + F, 24, 24, 1, 2, 2);                            \
    NAME->template layer_get<2>().init_layer(10 + F, 12, 12, 12 + F, 5, 5);                       \
    NAME->template layer_get<3>().init_layer(12 + F, 8, 8, 1, 2, 2);                              \
    NAME->template layer_get<4>().init_layer((12 + F) * 4 * 4, 500 + F);                          \
    NAME->template layer_get<5>().init_layer(500 + F, 10);                                        \
    NAME->template layer_get<0>().batch_size = 64;                                                \
    NAME->template layer_get<2>().batch_size = 64;                                                \
    NAME->template layer_get<4>().batch_size = 64;                                                \
    NAME->template layer_get<5>().batch_size = 64;                                                \
    NAME->pretrain(dataset.training_images, 10);                                                  \
    NAME->fine_tune(dataset.training_images, dataset.training_labels, 10);

    decl_dbn6(dbn1,dbn1_t,0)

    return 0;
}
