//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

#define decl_dbn6(NAME, NAME_T, F)                                                                                              \
    using NAME_T =                                                                                                              \
        dll::dyn_dbn_desc<                                                                                                      \
            dll::dbn_layers<                                                                                                    \
                dll::conv_rbm_square_desc<1, 28, 10 + F, 5, dll::momentum, dll::batch_size<64>>::layer_t,                       \
                dll::mp_3d_layer_desc<10 + F, 24, 24, 1, 2, 2>::layer_t,                                                        \
                dll::conv_rbm_square_desc<10 + F, 12, 12 + F, 5, dll::momentum, dll::batch_size<64>>::layer_t,                  \
                dll::mp_3d_layer_desc<12 + F, 8, 8, 1, 2, 2>::layer_t,                                                          \
                dll::rbm_desc<(12 + F) * 4 * 4, 500 + F, dll::momentum, dll::batch_size<64>>::layer_t,                          \
                dll::rbm_desc<500 + F, 10, dll::momentum, dll::batch_size<64>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>, \
            dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<64>>::dbn_t;             \
    auto NAME = std::make_unique<NAME_T>();                                                                                     \
    NAME->pretrain(dataset.training_images, 10);                                                                                \
    NAME->fine_tune(dataset.training_images, dataset.training_labels, 10);

    decl_dbn6(dbn1,dbn1_t,0)
    decl_dbn6(dbn2,dbn2_t,1)
    decl_dbn6(dbn3,dbn3_t,2)
    decl_dbn6(dbn4,dbn4_t,3)
    decl_dbn6(dbn5,dbn5_t,4)

    return 0;
}
