//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/rbm/conv_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

constexpr size_t EPOCHS = 5;

using clock      = std::chrono::steady_clock;
using time_point = std::chrono::time_point<clock>;
using resolution = std::chrono::milliseconds;

struct perf_timer {
    std::string name;
    size_t repeat;

    time_point start;

    perf_timer(std::string name, size_t repeat) : name(name), repeat(repeat) {
        start = clock::now();
    }

    ~perf_timer(){
        auto end      = clock::now();
        auto duration = std::chrono::duration_cast<resolution>(end - start).count();

        std::cout << name << ": " << duration / double(repeat) << "ms" << std::endl;
    }
};

#define MEASURE(rbm, name, data)                                                           \
    {                                                                                      \
        size_t d_min = std::numeric_limits<size_t>::max();                       \
        size_t d_max = 0;                                                             \
        for (size_t i = 0; i < EPOCHS; ++i) {                                         \
            time_point start = clock::now();                                               \
            rbm.train<false>(data, 1);                                                     \
            time_point end = clock::now();                                                 \
            size_t d  = std::chrono::duration_cast<resolution>(end - start).count();  \
            d_min          = std::min(d_min, d);                                           \
            d_max          = std::max(d_max, d);                                           \
        }                                                                                  \
        std::cout << name << ": min:" << d_min << "ms max:" << d_max << "ms" << std::endl; \
    }

} //end of anonymous namespace

int main(int argc, char* argv []) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>();
    dataset.training_images.resize(10000);

    std::string number;
    if(argc > 1){
        number = argv[1];
    }

    auto n = dataset.training_images.size();

    mnist::binarize_dataset(dataset);

    decltype(auto) data_1 = dataset.training_images;
    std::vector<etl::fast_dyn_matrix<float, 40, 20, 20>> data_2(n);
    std::vector<etl::fast_dyn_matrix<float, 40, 16, 16>> data_3(n);
    std::vector<etl::fast_dyn_matrix<float, 96, 12, 12>> data_4(n);

    for(size_t i = 0; i < n; ++i){
        data_2[i] = etl::normal_generator() * 255.0;
        data_3[i] = etl::normal_generator() * 255.0;
        data_4[i] = etl::normal_generator() * 255.0;
    }

    mnist::binarize_each(data_2);
    mnist::binarize_each(data_3);
    mnist::binarize_each(data_4);

    cpp_assert(data_1[0].size() == 784, "Invalid input size");
    cpp_assert(data_2[0].size() == 40 * 20 * 20, "Invalid input size");
    cpp_assert(data_3[0].size() == 40 * 16 * 16, "Invalid input size");
    cpp_assert(data_4[0].size() == 96 * 12 * 12, "Invalid input size");

    std::cout << n << " images used for training" << std::endl;
    std::cout << etl::threads << " maximum threads" << std::endl;

    if(number.empty() || number == "3"){
#define BATCH_MEASURE(batch)                                                                                        \
    {                                                                                                               \
        dll::conv_rbm_square_desc<1, 28, 40, 9, dll::batch_size<batch>, dll::weight_type<float>>::layer_t crbm_1;  \
        dll::conv_rbm_square_desc<40, 20, 40, 5, dll::batch_size<batch>, dll::weight_type<float>>::layer_t crbm_2; \
        dll::conv_rbm_square_desc<40, 16, 96, 5, dll::batch_size<batch>, dll::weight_type<float>>::layer_t crbm_3; \
        dll::conv_rbm_square_desc<96, 12, 8, 3, dll::batch_size<batch>, dll::weight_type<float>>::layer_t crbm_4;   \
        MEASURE(crbm_1, "crbm_1x28x28_batch_" #batch, data_1);                                                       \
        MEASURE(crbm_2, "crbm_40x20x20_batch_" #batch, data_2);                                                      \
        MEASURE(crbm_3, "crbm_40x16x16batch_" #batch, data_3);                                                       \
        MEASURE(crbm_4, "crbm_100x12x12_batch_" #batch, data_4);                                                     \
    }

        //BATCH_MEASURE(8);
        //BATCH_MEASURE(16);
        //BATCH_MEASURE(24);
        //BATCH_MEASURE(32);
        BATCH_MEASURE(64);
        //BATCH_MEASURE(128);
        //BATCH_MEASURE(256);
        //BATCH_MEASURE(512);

    }

    return 0;
}
