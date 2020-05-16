//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/rbm/rbm.hpp"

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

#define MEASURE(rbm, name, data)                                                     \
    {                                                                                \
        time_point start = clock::now();                                             \
        rbm.train<false>(data, EPOCHS);                                              \
        time_point end = clock::now();                                               \
        auto duration = std::chrono::duration_cast<resolution>(end - start).count(); \
        std::cout << name << ": " << duration / double(EPOCHS) << "ms" << std::endl; \
    }

} //end of anonymous namespace

int main(int argc, char* argv []) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();
    //dataset.training_images.resize(1000);

    std::string number;
    if(argc > 1){
        number = argv[1];
    }

    auto n = dataset.training_images.size();

    mnist::binarize_dataset(dataset);

    decltype(auto) data_1 = dataset.training_images;
    std::vector<etl::dyn_vector<float>> data_2(n, etl::dyn_vector<float>(500));
    std::vector<etl::dyn_vector<float>> data_3(n, etl::dyn_vector<float>(500));
    std::vector<etl::dyn_vector<float>> data_4(n, etl::dyn_vector<float>(2000));

    for(size_t i = 0; i < n; ++i){
        data_2[i] = etl::normal_generator() * 255.0;
        data_3[i] = etl::normal_generator() * 255.0;
        data_4[i] = etl::normal_generator() * 255.0;
    }

    mnist::binarize_each(data_2);
    mnist::binarize_each(data_3);
    mnist::binarize_each(data_4);

    cpp_assert(data_1[0].size() == 784, "Invalid input size");
    cpp_assert(data_2[0].size() == 500, "Invalid input size");
    cpp_assert(data_3[0].size() == 500, "Invalid input size");
    cpp_assert(data_4[0].size() == 2000, "Invalid input size");

    std::cout << n << " images used for training" << std::endl;

    if(number.empty() || number == "3"){
#define BATCH_MEASURE(batch)                                                                      \
    {                                                                                             \
        dll::rbm_desc<784, 500, dll::batch_size<batch>, dll::weight_type<float>>::layer_t rbm_1;  \
        dll::rbm_desc<500, 500, dll::batch_size<batch>, dll::weight_type<float>>::layer_t rbm_2;  \
        dll::rbm_desc<500, 2000, dll::batch_size<batch>, dll::weight_type<float>>::layer_t rbm_3; \
        dll::rbm_desc<2000, 10, dll::batch_size<batch>, dll::weight_type<float>>::layer_t rbm_4;  \
        MEASURE(rbm_1, "rbm_784_500_batch_" #batch, data_1);                                      \
        MEASURE(rbm_2, "rbm_500_500_batch_" #batch, data_2);                                      \
        MEASURE(rbm_3, "rbm_500_2000_batch_" #batch, data_3);                                     \
        MEASURE(rbm_4, "rbm_2000_10_batch_" #batch, data_4);                                      \
    }
        BATCH_MEASURE(8);
        BATCH_MEASURE(16);
        BATCH_MEASURE(24);
        BATCH_MEASURE(32);
        BATCH_MEASURE(64);
        BATCH_MEASURE(128);
        BATCH_MEASURE(256);
        BATCH_MEASURE(512);
    }

    return 0;
}
