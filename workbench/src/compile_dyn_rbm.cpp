//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/rbm/dyn_rbm.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// 5 3-layer networks

int main(int, char**) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();

    using dbn3_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_rbm_desc<dll::momentum>::layer_t,
                dll::dyn_rbm_desc<dll::momentum>::layer_t,
                dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
            dll::batch_size<64>, dll::trainer<dll::sgd_trainer>>::dbn_t;

#define decl_dbn3(NAME, F)                                      \
    auto NAME = std::make_unique<dbn3_t>();                     \
    NAME->template layer_get<0>().init_layer(28 * 28, 100 + F); \
    NAME->template layer_get<1>().init_layer(100 + F, 200 + F); \
    NAME->template layer_get<2>().init_layer(200 + F, 10);      \
    NAME->template layer_get<0>().batch_size = 64;              \
    NAME->template layer_get<1>().batch_size = 64;              \
    NAME->template layer_get<2>().batch_size = 64;              \
    NAME->pretrain(dataset.training_images, 20);                \
    NAME->fine_tune(dataset.training_images, dataset.training_labels, 20);

    decl_dbn3(dbn1,1)
    decl_dbn3(dbn2,2)
    decl_dbn3(dbn3,3)
    decl_dbn3(dbn4,4)
    decl_dbn3(dbn5,5)

    return 0;
}
