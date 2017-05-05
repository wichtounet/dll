//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/conv_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

constexpr std::size_t EPOCHS = 5;

using clock      = std::chrono::steady_clock;
using time_point = std::chrono::time_point<clock>;
using resolution = std::chrono::milliseconds;

#define MEASURE(rbm, name, data)                                                           \
    {                                                                                      \
        std::size_t d_min = std::numeric_limits<std::size_t>::max();                       \
        std::size_t d_max = 0;                                                             \
        for (std::size_t i = 0; i < EPOCHS; ++i) {                                         \
            time_point start = clock::now();                                               \
            rbm.train<false>(data, 1);                                                     \
            time_point end = clock::now();                                                 \
            std::size_t d  = std::chrono::duration_cast<resolution>(end - start).count();  \
            d_min          = std::min(d_min, d);                                           \
            d_max          = std::max(d_max, d);                                           \
        }                                                                                  \
        std::cout << name << ": min:" << d_min << "ms max:" << d_max << "ms" << std::endl; \
    }

} //end of anonymous namespace

int main(int argc, char* argv []) {
    auto dataset = mnist::read_dataset<std::vector, std::vector, float>(5000);

    std::string sub;
    if(argc > 1){
        sub = argv[1];
    }

    auto n = dataset.training_images.size();

    mnist::binarize_dataset(dataset);

    for (auto& image : dataset.training_images) {
        image.reserve(image.size() * 2);
        auto end = image.size();
        for (std::size_t i = 0; i < end; ++i) {
            image.push_back(image[i]);
        }
    }

    std::cout << n << " images used for training" << std::endl;
    std::cout << etl::threads << " maximum threads" << std::endl;

    if(sub.empty() || sub == "batch"){
        dll::conv_rbm_desc_square<2, 28, 40, 17, dll::batch_size<64>, dll::weight_type<float>>::layer_t crbm_1;
        MEASURE(crbm_1, "batch", dataset.training_images);
    }

    if(sub.empty() || sub == "parallel"){
        dll::conv_rbm_desc_square<2, 28, 40, 17, dll::batch_size<64>, dll::parallel_mode, dll::weight_type<float>>::layer_t crbm_2;
        MEASURE(crbm_2, "parallel", dataset.training_images);
    }

    if(!sub.empty()){
        dll::dump_timers();
    }

    return 0;
}
