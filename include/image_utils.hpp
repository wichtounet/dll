//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_IMAGE_UTILS_HPP
#define DBN_IMAGE_UTILS_HPP

template<typename T>
typename std::enable_if<!std::numeric_limits<T>::is_signed, void>::type binarize(std::vector<T>& values){
    for(auto& v : values){
        v = v > 10.0 ? 1.0 : 0.0;
    }
}

template<typename T>
typename std::enable_if<!std::numeric_limits<T>::is_signed, void>::type binarize_each(std::vector<std::vector<T>>& values){
    for(auto& v : values){
        binarize(v);
    }
}

#endif
