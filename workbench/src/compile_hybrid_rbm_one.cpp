//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/rbm/rbm.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// 1 3-layer networks

int main(int, char**) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();

    mnist::binarize_dataset(dataset);

#define decl_dbn3(NAME, NAME_T, F)                                                                                              \
    using NAME_T =                                                                                                              \
        dll::dyn_dbn_desc<                                                                                                      \
            dll::dbn_layers<                                                                                                    \
                dll::rbm_desc<28 * 28, 500 + F, dll::momentum, dll::batch_size<64>>::layer_t,                                   \
                dll::rbm_desc<500 + F, 400 + F, dll::momentum, dll::batch_size<64>>::layer_t,                                   \
                dll::rbm_desc<400 + F, 10, dll::momentum, dll::batch_size<64>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>, \
            dll::trainer<dll::sgd_trainer>, dll::batch_size<64>>::dbn_t;                                                        \
    auto NAME = std::make_unique<NAME_T>();                                                                                     \
    NAME->pretrain(dataset.training_images, 10);                                                                                \
    NAME->fine_tune(dataset.training_images, dataset.training_labels, 10);

    decl_dbn3(dbn1,dbn1_t,1)

    return 0;
}
