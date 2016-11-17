//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll/neural/dense_layer.hpp"
#include "dll/test.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

template <typename Dataset>
void mnist_scale(Dataset& dataset) {
    for (auto& image : dataset.training_images) {
        for (auto& pixel : image) {
            pixel *= (1.0 / 256.0);
        }
    }

    for (auto& image : dataset.test_images) {
        for (auto& pixel : image) {
            pixel *= (1.0 / 256.0);
        }
    }
}

template<typename D>
void basic_ae(const D& dataset){
    using network_t = dll::dbn_desc<dll::dbn_layers<
            dll::dense_desc<28 * 28, 200>::layer_t,
            dll::dense_desc<200, 28 * 28>::layer_t
        >, dll::momentum, dll::trainer<dll::sgd_trainer>, dll::batch_size<64>>::dbn_t;

    auto ae = std::make_unique<network_t>();

    ae->display();

    ae->learning_rate = 0.1;

    auto ft_error = ae->fine_tune_ae(dataset.training_images, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    auto test_error = dll::test_set_ae(*ae, dataset.test_images);
    std::cout << "test_error:" << test_error << std::endl;
}

} //end of anonymous namespace

int main(int /*argc*/, char* /*argv*/ []) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();
    dataset.training_images.resize(20000);

    auto n = dataset.training_images.size();
    std::cout << n << " samples to test" << std::endl;

    //mnist_scale(dataset);
    mnist::binarize_dataset(dataset);

    basic_ae(dataset);

    return 0;
}
