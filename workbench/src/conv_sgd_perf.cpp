//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#define ETL_COUNTERS

#include "dll/neural/conv/conv_layer.hpp"
#include "dll/neural/conv/conv_same_layer.hpp"
#include "dll/neural/dense/dense_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/test.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include "cifar/cifar10_reader.hpp"

namespace {

/*!
 * \brief Scale all values of a MNIST dataset into [0,1]
 */
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

void first_ex(){
    // First experiment : Conv -> Conv -> Dense -> Dense
    // Current speed on frigg:
    //   21 seconds (mkl-threads, default options)
    //   27-29 seconds (mkl, default options)
    //   40 seconds (mkl, conv4_prefer_blas)
    //   36 seconds (mkl-threads, conv4_prefer_blas)

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(3000);

    auto n = dataset.training_images.size();
    std::cout << n << " samples to test" << std::endl;

    mnist::binarize_dataset(dataset);

    // Clean slate
    etl::reset_counters();
    dll::reset_timers();

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 6, 5, 5>::layer_t,
            dll::conv_layer_desc<6, 24, 24, 6, 5, 5>::layer_t,
            dll::dense_layer_desc<6 * 20 * 20, 500>::layer_t,
            dll::dense_layer_desc<500, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dbn_t>();

    // Train the network for performance sake
    net->display();
    net->fine_tune(dataset.training_images, dataset.training_labels, 20);

    std::cout << "DLL Timers" << std::endl;
    dll::dump_timers_one();

    std::cout << "ETL Counters" << std::endl;
    etl::dump_counters();
}

void second_ex(){
    // Second experiment : Conv -> Pooling -> Conv -> Dense -> Dense
    // Current speed on frigg:
    //   12-13 seconds (mkl-threads, default-options)
    //   12-13 seconds (mkl, default-options)
    //   14 seconds (mkl, conv4_prefer_blas)
    //   19 seconds (mkl-threads, conv4_prefer_blas)

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(3000);

    auto n = dataset.training_images.size();
    std::cout << n << " samples to test" << std::endl;

    mnist::binarize_dataset(dataset);

    // Clean slate
    etl::reset_counters();
    dll::reset_timers();

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 10, 5, 5>::layer_t,
            dll::mp_2d_layer_desc<10, 24, 24, 2, 2>::layer_t,
            dll::conv_layer_desc<10, 12, 12, 10, 5, 5>::layer_t,
            dll::dense_layer_desc<10 * 8 * 8, 250>::layer_t,
            dll::dense_layer_desc<250, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dbn_t>();

    // Train the network for performance sake
    net->display();
    net->fine_tune(dataset.training_images, dataset.training_labels, 20);

    std::cout << "DLL Timers" << std::endl;
    dll::dump_timers_one();

    std::cout << "ETL Counters" << std::endl;
    etl::dump_counters();
}

void third_ex(){
    // Third experiment : Conv -> Pooling -> Conv -> Pooling -> Dense -> Dense
    // Current speed on frigg:
    //   24 seconds (mkl-threads, default-options)
    //   21 seconds (mkl, default-options)
    //   25 seconds (mkl, conv4_prefer_blas)
    //   38 seconds (mkl-threads, conv4_prefer_blas)

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(6000);

    auto n = dataset.training_images.size();
    std::cout << n << " samples to test" << std::endl;

    mnist_scale(dataset);

    // Clean slate
    etl::reset_counters();
    dll::reset_timers();

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<1, 28, 28, 10, 5, 5>::layer_t,
            dll::mp_2d_layer_desc<10, 24, 24, 2, 2>::layer_t,
            dll::conv_layer_desc<10, 12, 12, 10, 5, 5>::layer_t,
            dll::mp_2d_layer_desc<10, 8, 8, 2, 2>::layer_t,
            dll::dense_layer_desc<10 * 4 * 4, 300>::layer_t,
            dll::dense_layer_desc<300, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dbn_t>();

    net->learning_rate = 0.05;

    // Train the network for performance sake
    net->display();
    net->fine_tune(dataset.training_images, dataset.training_labels, 20);

    std::cout << "DLL Timers" << std::endl;
    dll::dump_timers_one();

    std::cout << "ETL Counters" << std::endl;
    etl::dump_counters();
}

void fourth_ex(){
    // Third experiment (CIFAR) : Conv -> Pooling -> Conv -> Pooling -> Dense -> Dense
    // This also uses momentum and RELU, more realistic
    // Current speed on frigg:
    //   146 seconds (mkl-threads, default-options)
    //   109 seconds (mkl, default-options)
    //   109 seconds (mkl, conv4_prefer_blas)
    //   177 seconds (mkl-threads, conv4_prefer_blas)

    auto dataset = cifar::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 3, 32, 32>>();

    // Clean slate
    etl::reset_counters();
    dll::reset_timers();

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<3, 32, 32, 12, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_2d_layer_desc<12, 28, 28, 2, 2>::layer_t,
            dll::conv_layer_desc<12, 14, 14, 24, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_2d_layer_desc<24, 12, 12, 2, 2>::layer_t,
            dll::dense_layer_desc<24 * 6 * 6, 64>::layer_t,
            dll::dense_layer_desc<64, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dbn_t>();

    net->learning_rate = 0.001;
    net->initial_momentum = 0.9;
    net->momentum = 0.9;
    net->goal = -1.0;

    // Train the network for performance sake
    net->display();
    net->fine_tune(dataset.training_images, dataset.training_labels, 5);

    std::cout << "DLL Timers" << std::endl;
    dll::dump_timers_one();

    std::cout << "ETL Counters" << std::endl;
    etl::dump_counters();
}

void fifth_ex(){
    // Third experiment (MNIST) : Conv -> Conv -> Pooling -> Conv -> Conv -> Pooling -> Dense -> Dense
    // This also uses momentum and RELU, more realistic
    // Current speed on frigg:
    //   50 seconds (mkl-threads, default-options)
    //   30-34 seconds (mkl, default-options)
    //   12-13 seconds (mkl, conv4_prefer_blas)
    //   27 seconds (mkl-threads, conv4_prefer_blas)

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(3000);

    mnist::binarize_dataset(dataset);

    // Clean slate
    etl::reset_counters();
    dll::reset_timers();

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_same_desc<1, 28, 28, 12, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::conv_same_desc<12, 28, 28, 12, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_2d_layer_desc<12, 28, 28, 2, 2>::layer_t,

            dll::conv_same_desc<12, 14, 14, 12, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::conv_same_desc<12, 14, 14, 12, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_2d_layer_desc<12, 14, 14, 2, 2>::layer_t,

            dll::dense_layer_desc<12 * 7 * 7, 64, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_layer_desc<64, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dbn_t>();

    net->learning_rate = 0.001;
    net->initial_momentum = 0.9;
    net->momentum = 0.9;
    net->goal = -1.0;

    // Train the network for performance sake
    net->display();
    net->fine_tune(dataset.training_images, dataset.training_labels, 5);

    std::cout << "DLL Timers" << std::endl;
    dll::dump_timers_one();

    std::cout << "ETL Counters" << std::endl;
    etl::dump_counters();
}

} // end of anonymous namespace

int main(int argc, char* argv []) {
    if(argc == 1){
        first_ex();
        return 0;
    }

    std::string select(argv[1]);
    if(select == "A"){
        first_ex();
    } else if(select == "B"){
        second_ex();
    } else if(select == "C"){
        third_ex();
    } else if(select == "D"){
        fourth_ex();
    } else if(select == "E"){
        fifth_ex();
    }

    return 0;
}
