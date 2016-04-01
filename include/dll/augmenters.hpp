//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <typename Augment>
struct augmenter;

template <std::size_t C>
struct augmenter <copy<C>> {
    template <typename Input, typename Output>
    static void apply(Output& result, const Input& input) {
        for(std::size_t c = 0; c < C; ++c){
            // Simply create a copy
            result.push_back(input);
        }
    }

    static void concat_name(std::string& name) {
        name += " copy<" + std::to_string(C) + ">";
    }
};

} //end of dll namespace
