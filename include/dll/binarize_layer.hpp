//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_BINARIZE_LAYER_INL
#define DLL_BINARIZE_LAYER_INL

#include "etl/etl.hpp"

namespace dll {

/*!
 * \brief Simple thresholding binarize layer
 */
template<typename Desc>
struct binarize_layer {
    using desc = Desc;

    static constexpr const std::size_t Threshold = desc::T;

    binarize_layer() = default;

    static std::string to_short_string(){
        return "Binarize";
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
        for(auto& value : h){
            value = value > Threshold ? 1 : 0;
        }
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
};

//Allow odr-use of the constexpr static members

template<typename Desc>
const std::size_t binarize_layer<Desc>::Threshold;

} //end of dll namespace

#endif
