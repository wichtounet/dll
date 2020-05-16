//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/rbm/dyn_rbm.hpp"
#include "dll/rbm/rbm.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

constexpr size_t EPOCHS = 10;

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

#define MEASURE(dbn, name, data, result)                                             \
    {                                                                                \
        time_point start = clock::now();                                             \
        dbn->pretrain(data, EPOCHS);                                                 \
        time_point end = clock::now();                                               \
        auto duration = std::chrono::duration_cast<resolution>(end - start).count(); \
        std::cout << name << ": " << duration << "ms" << std::endl;                  \
        result = duration;                                                           \
    }

} //end of anonymous namespace

int main(int, char**) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();
    dataset.training_images.resize(10000);

    auto n = dataset.training_images.size();

    mnist::binarize_dataset(dataset);

    std::cout << n << " images used for training" << std::endl;

    decltype(auto) data_1 = dataset.training_images;

    using dyn_dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_rbm_desc<dll::momentum, dll::batch_size<50>>::layer_t,
                dll::dyn_rbm_desc<dll::momentum, dll::batch_size<50>>::layer_t,
                dll::dyn_rbm_desc<dll::momentum, dll::batch_size<50>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>
        , dll::watcher<dll::mute_dbn_watcher>>::dbn_t;

    auto dyn_dbn = std::make_unique<dyn_dbn_t>();

    dyn_dbn->template layer_get<0>().init_layer(28 * 28, 250);
    dyn_dbn->template layer_get<1>().init_layer(250, 500);
    dyn_dbn->template layer_get<2>().init_layer(500, 10);

    size_t dyn_duration = 0;
    MEASURE(dyn_dbn, "dyn_dbn_pretrain", data_1, dyn_duration);

    dll::dump_timers();
    dll::reset_timers();

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 250, dll::momentum, dll::batch_size<50>>::layer_t,
            dll::rbm_desc<250, 500,     dll::momentum, dll::batch_size<50>>::layer_t,
            dll::rbm_desc<500, 10,      dll::momentum, dll::batch_size<50>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>
        , dll::watcher<dll::mute_dbn_watcher>>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    size_t static_duration = 0;
    MEASURE(dbn, "dbn_pretrain", data_1, static_duration);

    dll::dump_timers();

    std::cout << "Ratio:" << 100.0 * (double(static_duration) / double(dyn_duration)) << std::endl;

    return 0;
}
