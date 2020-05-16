//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include <fenv.h>

#include "dll/rbm/conv_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

constexpr size_t EPOCHS = 10;

using clock      = std::chrono::steady_clock;
using time_point = std::chrono::time_point<clock>;
using resolution = std::chrono::milliseconds;

#define MEASURE(rbm, name, data)                                                                                                    \
    {                                                                                                                               \
        size_t d_min  = std::numeric_limits<size_t>::max();                                                               \
        size_t d_max  = 0;                                                                                                     \
        size_t d_mean = 0;                                                                                                     \
        for (size_t i = 0; i < EPOCHS; ++i) {                                                                                  \
            time_point start = clock::now();                                                                                        \
            rbm.train<false>(data, 1);                                                                                              \
            time_point end = clock::now();                                                                                          \
            size_t d  = std::chrono::duration_cast<resolution>(end - start).count();                                           \
            d_min          = std::min(d_min, d);                                                                                    \
            d_max          = std::max(d_max, d);                                                                                    \
            d_mean         = d_mean + d;                                                                                            \
        }                                                                                                                           \
        std::cout << name << ": min:" << d_min << "ms max:" << d_max << "ms mean:" << d_mean / double(EPOCHS) << "ms" << std::endl; \
    }

} //end of anonymous namespace

int main(int argc, char* argv []) {
    // With this flag, float version is much faster
    // _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(2500);

    std::string sub;
    if(argc > 1){
        sub = argv[1];
    }

    auto n = dataset.training_images.size();

    mnist::normalize_dataset(dataset);

    std::cout << n << " images used for training" << std::endl;
    std::cout << etl::threads << " maximum threads" << std::endl;

    if(sub.empty() || sub == "batch"){
        dll::conv_rbm_square_desc<1, 28, 20, 17, dll::visible<dll::unit_type::GAUSSIAN>, dll::weight_decay<dll::decay_type::L2>, dll::momentum, dll::shuffle, dll::batch_size<25>, dll::weight_type<float>>::layer_t crbm_float;
        dll::conv_rbm_square_desc<1, 28, 20, 17, dll::visible<dll::unit_type::GAUSSIAN>, dll::weight_decay<dll::decay_type::L2>, dll::momentum, dll::shuffle, dll::batch_size<25>, dll::weight_type<double>>::layer_t crbm_double;
        MEASURE(crbm_float, "batch_float", dataset.training_images);
        MEASURE(crbm_double, "batch_double", dataset.training_images);
    }

    if(!sub.empty()){
        dll::dump_timers();
    }

    return 0;
}
