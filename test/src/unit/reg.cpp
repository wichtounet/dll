//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/neural/rnn/rnn_layer.hpp"
#include "dll/neural/recurrent/recurrent_last_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

namespace {

void generate(std::vector<etl::fast_dyn_matrix<float, 3>>& samples, std::vector<size_t>& labels, size_t N){
    std::random_device rd;
    std::mt19937_64 engine(rd());

    std::uniform_int_distribution<int> dist(0, 25);

    for(size_t i = 0; i < N; ++i){
        samples.emplace_back();

        auto& back = samples.back();

        back[0] = dist(engine) / 25.0f;
        back[1] = dist(engine) / 25.0f;
        back[2] = dist(engine) / 25.0f;

        labels.push_back((back[0] + back[1] + back[2]) / 3.0f);
    }
}

} // end of anonymous namespace

DLL_TEST_CASE("unit/reg/1", "[unit][reg]") {
    std::vector<etl::fast_dyn_matrix<float, 3>> samples;
    std::vector<size_t> labels;

    generate(samples, labels, 1000);

    using network_t = dll::dyn_network_desc<
        dll::network_layers<
            dll::dense_layer<3, 1, dll::tanh>
        >
        , dll::mean_squared_error
        , dll::batch_size<10>
        , dll::adadelta
    >::network_t;

    auto net = std::make_unique<network_t>();

    REQUIRE(net->fine_tune_reg(samples, labels, 30) < 0.15);

    // Mostly here for compilation's sake
    net->evaluate_reg(samples, labels);

    REQUIRE(net->evaluate_error_reg(samples, labels) < 0.25);
}

DLL_TEST_CASE("unit/reg/2", "[unit][reg]") {
    std::vector<etl::fast_dyn_matrix<float, 3>> samples;
    std::vector<size_t> labels;

    generate(samples, labels, 1000);

    using network_t = dll::dyn_network_desc<
        dll::network_layers<
            dll::dense_layer<3, 1, dll::tanh>
        >
        , dll::mean_squared_error
        , dll::batch_size<10>
        , dll::adadelta
        , dll::shuffle
    >::network_t;

    auto net = std::make_unique<network_t>();

    REQUIRE(net->fine_tune_reg(samples, labels, 30) < 0.15);

    // Mostly here for compilation's sake
    net->evaluate_reg(samples, labels);

    REQUIRE(net->evaluate_error_reg(samples, labels) < 0.25);
}

DLL_TEST_CASE("unit/reg/3", "[unit][reg]") {
    std::vector<etl::fast_dyn_matrix<float, 3>> samples;
    std::vector<size_t> labels;

    generate(samples, labels, 5000);

    using network_t = dll::dyn_network_desc<
        dll::network_layers<
            dll::dense_layer<3, 10, dll::tanh>,
            dll::dense_layer<10, 1, dll::tanh>
        >
        , dll::mean_squared_error
        , dll::batch_size<10>
        , dll::adadelta
        , dll::shuffle
    >::network_t;

    auto net = std::make_unique<network_t>();

    net->display_pretty();

    REQUIRE(net->fine_tune_reg(samples, labels, 30) < 0.15);

    // Mostly here for compilation's sake
    net->evaluate_reg(samples, labels);

    REQUIRE(net->evaluate_error_reg(samples, labels) < 0.25);
}
