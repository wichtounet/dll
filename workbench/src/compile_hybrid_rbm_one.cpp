//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

    using dbn1_t =
        dll::dyn_dbn_desc<
            dll::dbn_layers<
                dll::rbm_desc<28*28, 100, dll::momentum>::layer_t,
                dll::rbm_desc<100, 200, dll::momentum>::layer_t,
                dll::rbm_desc<200, 10, dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>>::dbn_t;
    auto dbn1 = std::make_unique<dbn1_t>();
    dbn1->display();
    dbn1->pretrain(dataset.training_images, 20);

    return 0;
}
