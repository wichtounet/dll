//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/rbm/rbm.hpp"
#include "dll/test.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

template<typename D>
void rbm_dae(const D& dataset){
    std::cout << " Test RBM Denoising Auto-Encoder" << std::endl;

    using network_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>>::layer_t
        >,
        dll::batch_size<50>>::dbn_t;

    auto ae = std::make_unique<network_t>();

    ae->display();

    ae->template layer_get<0>().learning_rate = 0.001;
    ae->template layer_get<0>().initial_momentum = 0.9;

    ae->pretrain_denoising_auto(dataset.training_images, 50, 0.3);
}

} //end of anonymous namespace

int main(int /*argc*/, char* /*argv*/ []) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();
    dataset.training_images.resize(20000);

    auto n = dataset.training_images.size();
    std::cout << n << " samples to test" << std::endl;

    mnist::binarize_dataset(dataset);
    rbm_dae(dataset);

    return 0;
}
