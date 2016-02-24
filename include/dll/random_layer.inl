//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "neural_base.hpp"

namespace dll {

template <typename Desc>
struct random_layer : neural_base<random_layer<Desc>> {
    using desc = Desc;

    random_layer() = default;

    static std::string to_short_string() {
        return "Random";
    }

    static void display() {
        std::cout << to_short_string() << std::endl;
    }

    template <typename Input, typename Output>
    static void activate_hidden(Output& output, const Input&) {
        output = etl::normal_generator<etl::value_t<Input>>();
    }

    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input&) {
        output = etl::normal_generator<etl::value_t<Input>>();
    }

    template <typename I, typename O_A>
    static void activate_many(const I& input, O_A& h_a) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            activate_one(input[i], h_a[i]);
        }
    }

    template <typename Input>
    static std::vector<Input> prepare_output(std::size_t samples) {
        return std::vector<Input>(samples);
    }

    template <typename Input>
    static Input prepare_one_output() {
        return {};
    }
};

} //end of dll namespace
