//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "neural_base.hpp"

namespace dll {

template<typename Desc>
struct rectifier_layer : neural_base<rectifier_layer<Desc>> {
    using desc = Desc;

    static constexpr const rectifier_method method = desc::method;

    static_assert(method == rectifier_method::ABS, "Only ABS rectifier has been implemented");

    rectifier_layer() = default;

    static std::string to_short_string(){
        return "Rectifier";
    }

    static void display(){
        std::cout << to_short_string() << std::endl;
    }

    template<typename Input, typename Output>
    static void activate_hidden(Output& output, const Input& input){
        if(method == rectifier_method::ABS){
            output = etl::abs(input);
        }
    }

    template<typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input){
        if(method == rectifier_method::ABS){
            output = etl::abs(input);
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
const rectifier_method rectifier_layer<Desc>::method;

} //end of dll namespace
