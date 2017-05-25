//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <atomic>
#include <thread>

#include "cpp_utils/data.hpp"

namespace dll {

template<typename Desc, typename Enable = void>
struct pre_scaler;

template<typename Desc>
struct pre_scaler <Desc, std::enable_if_t<Desc::ScalePre>> {
    static constexpr size_t S = Desc::ScalePre;

    template<typename O>
    static void transform(O&& target){
        target /= S;
    }
};

template<typename Desc>
struct pre_scaler <Desc, std::enable_if_t<!Desc::ScalePre>> {
    template<typename O>
    static void transform(O&& target){
        cpp_unused(target);
    }
};

template<typename Desc, typename Enable = void>
struct pre_binarizer;

template<typename Desc>
struct pre_binarizer <Desc, std::enable_if_t<Desc::BinarizePre>> {
    static constexpr size_t B = Desc::BinarizePre;

    template<typename O>
    static void transform(O&& target){
        for(auto& x : target){
            x = x > B ? 1.0 : 0.0;
        }
    }
};

template<typename Desc>
struct pre_binarizer <Desc, std::enable_if_t<!Desc::BinarizePre>> {
    template<typename O>
    static void transform(O&& target){
        cpp_unused(target);
    }
};

template<typename Desc, typename Enable = void>
struct pre_normalizer;

template<typename Desc>
struct pre_normalizer <Desc, std::enable_if_t<Desc::NormalizePre>> {
    template<typename O>
    static void transform(O&& target){
        cpp::normalize(target);
    }
};

template<typename Desc>
struct pre_normalizer <Desc, std::enable_if_t<!Desc::NormalizePre>> {
    template<typename O>
    static void transform(O&& target){
        cpp_unused(target);
    }
};

} //end of dll namespace
