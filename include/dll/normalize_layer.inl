//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_NORMALIZE_LAYER_INL
#define DLL_NORMALIZE_LAYER_INL

#include "cpp_utils/data.hpp"

namespace dll {

/*!
 * \brief Simple thresholding normalize layer
 */
template<typename Desc>
struct normalize_layer {
    using desc = Desc;

    normalize_layer() = default;

    static std::string to_short_string(){
        return "normalize";
    }

    static void display(){
        std::cout << to_short_string() << std::endl;
    }

    //TODO Ideally, the dbn should guess if h_a/h_s are used or only h_a
    template<typename I, typename O_A>
    static void activate_one(const I& v, O_A& h){
        activate_one(v, h, h);
    }

    template<typename I, typename O_A, typename O_S>
    static void activate_one(const I& v, O_A& h, O_S& /*h_s*/){
        h = v;
        cpp::normalize(h);
    }

    template<typename I, typename O_A, typename O_S>
    static void activate_many(const I& input, O_A& h_a, O_S& h_s){
        for(std::size_t i = 0; i < input.size(); ++i){
            activate_one(input[i], h_a[i], h_s[i]);
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

} //end of dll namespace

#endif
