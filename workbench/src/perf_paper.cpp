//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

constexpr const std::size_t EPOCHS = 5;

struct perf_timer {
    std::string name;
    std::size_t repeat;

    std::chrono::time_point<std::chrono::steady_clock> start;

    perf_timer(std::string name, std::size_t repeat) : name(name), repeat(repeat) {
        start = std::chrono::steady_clock::now();
    }

    ~perf_timer(){
        auto end      = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << name << ": " << duration / double(repeat) << "ms" << std::endl;
    }
};

} //end of anonymous namespace

int main(int argc, char* argv []) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();
    dataset.training_images.resize(1000);

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

    for(std::size_t i = 0; i < n; ++i){
        data_2[i] = etl::normal_generator();
        data_3[i] = etl::normal_generator();
        data_4[i] = etl::normal_generator();
    }

    cpp_assert(data_1[0].size() == 784, "Invalid input size");
    cpp_assert(data_2[0].size() == 500, "Invalid input size");
    cpp_assert(data_3[0].size() == 500, "Invalid input size");
    cpp_assert(data_4[0].size() == 2000, "Invalid input size");

    std::cout << n << " images used for training" << std::endl;

    if(number.empty() || number == "1"){
        dll::rbm_desc<784, 500, /*dll::parallel_mode, dll::serial,*/ dll::batch_size<1>, dll::weight_type<float>>::layer_t rbm_1;
        dll::rbm_desc<500, 500, /*dll::parallel_mode, dll::serial,*/ dll::batch_size<1>, dll::weight_type<float>>::layer_t rbm_2;
        dll::rbm_desc<500, 2000, /*dll::parallel_mode, dll::serial,*/ dll::batch_size<1>, dll::weight_type<float>>::layer_t rbm_3;
        dll::rbm_desc<2000, 10, /*dll::parallel_mode, dll::serial,*/ dll::batch_size<1>, dll::weight_type<float>>::layer_t rbm_4;

        {
            perf_timer timer("rbm_784_500_normal", EPOCHS);
            rbm_1.train<false>(data_1, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_500_normal", EPOCHS);
            rbm_2.train<false>(data_2, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_2000_normal", EPOCHS);
            rbm_3.train<false>(data_3, EPOCHS);
        }

        {
            perf_timer timer("rbm_2000_10_normal", EPOCHS);
            rbm_4.train<false>(data_4, EPOCHS);
        }
    }

    if(number.empty() || number == "2"){
        dll::rbm_desc<784, 500, dll::parallel_mode, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_1;
        dll::rbm_desc<500, 500, dll::parallel_mode, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_2;
        dll::rbm_desc<500, 2000, dll::parallel_mode, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_3;
        dll::rbm_desc<2000, 10, dll::parallel_mode, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_4;

        {
            perf_timer timer("rbm_784_500_par_64", EPOCHS);
            rbm_1.train<false>(data_1, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_500_par_64", EPOCHS);
            rbm_2.train<false>(data_2, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_2000_par_64", EPOCHS);
            rbm_3.train<false>(data_3, EPOCHS);
        }

        {
            perf_timer timer("rbm_2000_10_par_64", EPOCHS);
            rbm_4.train<false>(data_4, EPOCHS);
        }
    }

    if(number.empty() || number == "3"){
        dll::rbm_desc<784, 500, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_1;
        dll::rbm_desc<500, 500, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_2;
        dll::rbm_desc<500, 2000, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_3;
        dll::rbm_desc<2000, 10, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_4;

        {
            perf_timer timer("rbm_784_500_batch_64", EPOCHS);
            rbm_1.train<false>(data_1, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_500_batch_64", EPOCHS);
            rbm_2.train<false>(data_2, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_2000_batch_64", EPOCHS);
            rbm_3.train<false>(data_3, EPOCHS);
        }

        {
            perf_timer timer("rbm_2000_10_batch_64", EPOCHS);
            rbm_4.train<false>(data_4, EPOCHS);
        }
    }

    return 0;
}
