//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_IMAGE_UTILS_HPP
#define DBN_IMAGE_UTILS_HPP

template<typename Container>
void binarize_each(Container& values){
    for(auto& vec : values){
        for(auto& v : vec){
            v = v > 10.0 ? 1.0 : 0.0;
        }
    }
}

#endif
