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

void first_ex(){
    // First experiment : Conv -> Conv -> Dense -> Dense
    // Current training speed on frigg:
    //  ~100 seconds (mkl, default options)

    constexpr size_t N = 4096;
    constexpr size_t B = 128;

    std::vector<etl::fast_dyn_matrix<float, 3, 254, 254>> training_images;
    std::vector<size_t> training_labels;

    training_images.reserve(N);
    training_labels.reserve(N);

    for(size_t i = 0; i < N; ++i){
        training_images.emplace_back();
        training_labels.push_back(i % 1000);

        training_images.back() = etl::normal_generator();
    }

    auto n = training_images.size();
    std::cout << n << " samples to test" << std::endl;

    // Clean slate
    etl::reset_counters();
    dll::reset_timers();

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer_desc<3, 254, 254, 10, 3, 3>::layer_t,
            dll::mp_2d_layer_desc<10, 252, 252, 2, 2>::layer_t,

            dll::conv_layer_desc<10, 126, 126, 10, 3, 3>::layer_t,
            dll::mp_2d_layer_desc<10, 124, 124, 2, 2>::layer_t,

            dll::conv_layer_desc<10, 62, 62, 10, 3, 3>::layer_t,
            dll::mp_2d_layer_desc<10, 60, 60, 2, 2>::layer_t,

            dll::conv_layer_desc<10, 30, 30, 10, 3, 3>::layer_t,
            dll::mp_2d_layer_desc<10, 28, 28, 2, 2>::layer_t,

            dll::conv_layer_desc<10, 14, 14, 10, 3, 3>::layer_t,
            dll::mp_2d_layer_desc<10, 12, 12, 2, 2>::layer_t,

            dll::dense_layer_desc<10 * 6 * 6, 500>::layer_t,
            dll::dense_layer_desc<500, 1000, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::batch_mode, dll::verbose, dll::big_batch_size<5>, dll::batch_size<B>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dbn_t>();

    // Train the network for performance sake
    net->display();
    net->fine_tune(training_images, training_labels, 1);

    std::cout << "DLL Timers" << std::endl;
    dll::dump_timers_one();

    std::cout << "ETL Counters" << std::endl;
    etl::dump_counters();
}

void second_ex(){
    // Second experiment : Conv -> Conv -> Dense -> Dense
    // Current training speed on frigg:
    //  ~185 seconds (mkl, default options)

    constexpr size_t N = 4096;
    constexpr size_t B = 128;

    std::vector<etl::fast_dyn_matrix<float, 3, 256, 256>> training_images;
    std::vector<size_t> training_labels;

    training_images.reserve(N);
    training_labels.reserve(N);

    for(size_t i = 0; i < N; ++i){
        training_images.emplace_back();
        training_labels.push_back(i % 1000);

        training_images.back() = etl::normal_generator();
    }

    auto n = training_images.size();
    std::cout << n << " samples to test" << std::endl;

    // Clean slate
    etl::reset_counters();
    dll::reset_timers();

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_same_desc<3, 256, 256, 16, 3, 3>::layer_t,
            dll::mp_2d_layer_desc<16, 256, 256, 2, 2>::layer_t,

            dll::conv_same_desc<16, 128, 128, 16, 3, 3>::layer_t,
            dll::mp_2d_layer_desc<16, 128, 128, 2, 2>::layer_t,

            dll::conv_same_desc<16, 64, 64, 32, 3, 3>::layer_t,
            dll::mp_2d_layer_desc<32, 64, 64, 2, 2>::layer_t,

            dll::conv_same_desc<32, 32, 32, 32, 3, 3>::layer_t,
            dll::mp_2d_layer_desc<32, 32, 32, 2, 2>::layer_t,

            dll::conv_same_desc<32, 16, 16, 32, 3, 3>::layer_t,
            dll::mp_2d_layer_desc<32, 16, 16, 2, 2>::layer_t,

            dll::dense_layer_desc<32 * 8 * 8, 2048>::layer_t,
            dll::dense_layer_desc<2048, 1000, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::updater<dll::updater_type::MOMENTUM>, dll::verbose, dll::batch_mode, dll::big_batch_size<5>, dll::batch_size<B>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto net = std::make_unique<dbn_t>();

    // Train the network for performance sake
    net->display();
    net->fine_tune(training_images, training_labels, 1);

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
    }

    return 0;
}
