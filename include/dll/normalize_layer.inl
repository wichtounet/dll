//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_NORMALIZE_LAYER_INL
#define DLL_NORMALIZE_LAYER_INL

#include "cpp_utils/data.hpp"

#include "neural_base.hpp"

namespace dll {

/*!
 * \brief Simple thresholding normalize layer
 */
template <typename Desc>
struct normalize_layer : neural_base<normalize_layer<Desc>> {
    using desc = Desc;

    normalize_layer() = default;

    static std::string to_short_string() {
        return "normalize";
    }

    static void display() {
        std::cout << to_short_string() << std::endl;
    }

    template <typename Input, typename Output>
    static void activate_hidden(Output& output, const Input& input) {
        output = input;
        cpp::normalize(output);
    }

    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        output = input;
        cpp::normalize(output);
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

#endif
