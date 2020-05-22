//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll/neural/conv/conv_layer.hpp"
#include "dll/neural/dense/dense_layer.hpp"
#include "dll/neural/activation/activation_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/pooling/avgp_layer.hpp"
#include "dll/test.hpp"
#include "dll/dbn.hpp"

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

template<typename Dataset>
void dense_sgd(Dataset& dataset){
    using dense_net = dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 200, dll::activation<dll::function::SIGMOID>>::layer_t,
            dll::dense_layer_desc<200, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<50>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dense_net>();

    net->learning_rate = 0.1;
    net->momentum = 0.9;
    net->initial_momentum = 0.9;

    net->display();

    auto ft_error = net->fine_tune(dataset.training_images, dataset.training_labels, 10);
    std::cout << "ft_error:" << ft_error << std::endl;
}

template<typename Dataset>
void dense_sgd_split(Dataset& dataset){
    using dense_net = dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 200, dll::activation<dll::function::IDENTITY>>::layer_t,
            dll::activation_layer_desc<dll::function::SIGMOID>::layer_t,
            dll::dense_layer_desc<200, 10, dll::activation<dll::function::IDENTITY>>::layer_t,
            dll::activation_layer_desc<dll::function::SOFTMAX>::layer_t
        >,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<50>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dense_net>();

    net->learning_rate = 0.1;
    net->momentum = 0.9;
    net->initial_momentum = 0.9;

    net->display();

    auto ft_error = net->fine_tune(dataset.training_images, dataset.training_labels, 20);
    std::cout << "ft_error:" << ft_error << std::endl;
}

template<typename Dataset>
void conv_sgd(Dataset& dataset){
    using dense_net = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 4, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::conv_layer_desc<4, 24, 24, 4, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<4 * 20 * 20, 200, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<200, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<50>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dense_net>();

    net->learning_rate = 0.05;
    net->momentum = 0.9;
    net->initial_momentum = 0.9;

    net->display();

    auto ft_error = net->fine_tune(dataset.training_images, dataset.training_labels, 20);
    std::cout << "ft_error:" << ft_error << std::endl;
}

template<typename Dataset>
void conv_mp_sgd(Dataset& dataset){
    using dense_net = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 5, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_3d_layer_desc<5, 24, 24, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_layer_desc<5, 12, 12, 5, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_3d_layer_desc<5, 8, 8, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::dense_layer_desc<5 * 4 * 4, 200, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<200, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<50>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dense_net>();

    net->learning_rate = 0.05;
    net->momentum = 0.9;
    net->initial_momentum = 0.9;

    net->display();

    auto ft_error = net->fine_tune(dataset.training_images, dataset.training_labels, 20);
    std::cout << "ft_error:" << ft_error << std::endl;
}

template<typename Dataset>
void conv_avgp_sgd(Dataset& dataset){
    using dense_net = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 5, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::avgp_3d_layer_desc<5, 24, 24, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::conv_layer_desc<5, 12, 12, 5, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::avgp_3d_layer_desc<5, 8, 8, 1, 2, 2, dll::weight_type<float>>::layer_t,
            dll::dense_layer_desc<5 * 4 * 4, 200, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<200, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<50>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dense_net>();

    net->learning_rate = 0.05;
    net->momentum = 0.9;
    net->initial_momentum = 0.9;

    net->display();

    auto ft_error = net->fine_tune(dataset.training_images, dataset.training_labels, 20);
    std::cout << "ft_error:" << ft_error << std::endl;
}

} // end of anonymous namespace

int main(int /*argc*/, char* /*argv*/ []) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();
    dataset.training_images.resize(10000);
    dataset.training_labels.resize(10000);

    auto n = dataset.training_images.size();
    std::cout << n << " samples to test" << std::endl;

    mnist_scale(dataset);

    dense_sgd(dataset);
    dense_sgd_split(dataset);
    conv_sgd(dataset);
    conv_mp_sgd(dataset);
    conv_avgp_sgd(dataset);

    return 0;
}
