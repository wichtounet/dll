//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_BINARIZE_LAYER_INL
#define DLL_BINARIZE_LAYER_INL

#include "neural_base.hpp"

namespace dll {

/*!
 * \brief Simple thresholding binarize layer
 */
template<typename Desc>
struct binarize_layer : neural_base<binarize_layer<Desc>> {
    using desc = Desc;

    static constexpr const std::size_t Threshold = desc::T;

    binarize_layer() = default;

    static std::string to_short_string(){
        return "Binarize";
    }

    static void display(){
        std::cout << to_short_string() << std::endl;
    }

    template<typename Input, typename Output>
    static void activate_hidden(Output& output, const Input& input){
        output = input;

        for(auto& value : output){
            value = value > Threshold ? 1 : 0;
        }
    }

    template<typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input){
        output = input;

        for(auto& value : output){
            value = value > Threshold ? 1 : 0;
        }
    }

    template<typename I, typename O_A>
    static void activate_many(const I& input, O_A& h_a){
        for(std::size_t i = 0; i < input.size(); ++i){
            activate_one(input[i], h_a[i]);
        }
    }

    template<typename Input>
    static std::vector<Input> prepare_output(std::size_t samples){
        return std::vector<Input>(samples);
    }

    template<typename Input>
    static Input prepare_one_output(){
        return {};
    }
};

//Allow odr-use of the constexpr static members

template<typename Desc>
const std::size_t binarize_layer<Desc>::Threshold;

} //end of dll namespace

#endif
