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

    std::string number;
    if(argc > 1){
        number = argv[1];
    }

    mnist::binarize_dataset(dataset);

    dataset.training_images.resize(1000);

    std::cout << dataset.training_images.size() << " images used for training" << std::endl;

    if(number.empty() || number == "1"){
        dll::rbm_desc<784, 500, dll::parallel_mode, dll::serial, dll::weight_type<float>>::layer_t rbm_1;
        dll::rbm_desc<500, 500, dll::parallel_mode, dll::serial, dll::weight_type<float>>::layer_t rbm_2;
        dll::rbm_desc<500, 2000, dll::parallel_mode, dll::serial, dll::weight_type<float>>::layer_t rbm_3;
        dll::rbm_desc<2000, 10, dll::parallel_mode, dll::serial, dll::weight_type<float>>::layer_t rbm_4;

        {
            perf_timer timer("rbm_784_500_normal", EPOCHS);
            rbm_1.train<false>(dataset.training_images, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_500_normal", EPOCHS);
            rbm_2.train<false>(dataset.training_images, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_2000_normal", EPOCHS);
            rbm_3.train<false>(dataset.training_images, EPOCHS);
        }

        {
            perf_timer timer("rbm_2000_10_normal", EPOCHS);
            rbm_4.train<false>(dataset.training_images, EPOCHS);
        }
    }

    if(number.empty() || number == "2"){
        dll::rbm_desc<784, 500, dll::parallel_mode, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_1;
        dll::rbm_desc<500, 500, dll::parallel_mode, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_2;
        dll::rbm_desc<500, 2000, dll::parallel_mode, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_3;
        dll::rbm_desc<2000, 10, dll::parallel_mode, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_4;

        {
            perf_timer timer("rbm_784_500_par_64", EPOCHS);
            rbm_1.train<false>(dataset.training_images, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_500_par_64", EPOCHS);
            rbm_2.train<false>(dataset.training_images, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_2000_par_64", EPOCHS);
            rbm_3.train<false>(dataset.training_images, EPOCHS);
        }

        {
            perf_timer timer("rbm_2000_10_par_64", EPOCHS);
            rbm_4.train<false>(dataset.training_images, EPOCHS);
        }
    }

    if(number.empty() || number == "3"){
        dll::rbm_desc<784, 500, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_1;
        dll::rbm_desc<500, 500, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_2;
        dll::rbm_desc<500, 2000, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_3;
        dll::rbm_desc<2000, 10, dll::batch_size<64>, dll::weight_type<float>>::layer_t rbm_4;

        {
            perf_timer timer("rbm_784_500_batch_64", EPOCHS);
            rbm_1.train<false>(dataset.training_images, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_500_batch_64", EPOCHS);
            rbm_2.train<false>(dataset.training_images, EPOCHS);
        }

        {
            perf_timer timer("rbm_500_2000_batch_64", EPOCHS);
            rbm_3.train<false>(dataset.training_images, EPOCHS);
        }

        {
            perf_timer timer("rbm_2000_10_batch_64", EPOCHS);
            rbm_4.train<false>(dataset.training_images, EPOCHS);
        }
    }

    return 0;
}
