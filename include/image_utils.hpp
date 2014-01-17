//=======================================================================
// Copyright Baptiste Wicht 2014.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#ifndef DBN_IMAGE_UTILS_HPP
#define DBN_IMAGE_UTILS_HPP

template<typename T>
typename std::enable_if<!std::numeric_limits<T>::is_signed, void>::type binarize(std::vector<T>& values){
    auto middle = std::numeric_limits<T>::max() / 2;

    for(auto& v : values){
        v = v > 10 ? 1 : 0;
    }
}

template<typename T>
typename std::enable_if<!std::numeric_limits<T>::is_signed, void>::type binarize_each(std::vector<std::vector<T>>& values){
    for(auto& v : values){
        binarize(v);
    }
}

#endif
