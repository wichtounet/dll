//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/tmp.hpp"

#include <list>
#include <vector>
#include <deque>

namespace dll {

#ifdef DEBUG_CONVERT
#define debug_convert(X) std::cout << X << std::endl;
#else
#define debug_convert(X)
#endif

template<typename From, typename To>
struct cannot_convert {
    static constexpr const bool value = false;
};

template<typename From, typename To>
struct converter_one {
    static_assert(cannot_convert<From, To>::value, "DLL does not know how to convert your input type (one)");
};

// No conversion
// This should only happen when used from converter_many
template<typename From>
struct converter_one <From, From> {
    template<typename L>
    static const From& convert(const L&, const From& from){
        debug_convert("No convert");
        return from;
    }
};

// Convert a vector<T_F> to a dyn_vector<T_T>
template<typename T_F, typename T_T, typename A>
struct converter_one<std::vector<T_F, A>, etl::dyn_matrix<T_T, 1>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::dyn_matrix<T_T, 1> convert(const L&, const std::vector<T_F, A>& from){
        debug_convert("convert_one");
        return etl::dyn_matrix<T_T, 1>(from);
    }
};

// Convert a list<T_F> to a dyn_vector<T_T>
template<typename T_F, typename T_T, typename A>
struct converter_one<std::list<T_F, A>, etl::dyn_matrix<T_T, 1>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::dyn_matrix<T_T, 1> convert(const L&, const std::list<T_F, A>& from){
        debug_convert("convert_one");
        return etl::dyn_matrix<T_T, 1>(from);
    }
};

// Convert a deque<T_F> to a dyn_vector<T_T>
template<typename T_F, typename T_T, typename A>
struct converter_one<std::deque<T_F, A>, etl::dyn_matrix<T_T, 1>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::dyn_matrix<T_T, 1> convert(const L&, const std::deque<T_F, A>& from){
        debug_convert("convert_one");
        return etl::dyn_matrix<T_T, 1>(from);
    }
};

// Convert a vector<T_F> to a dyn_matrix<T_T, D>
template<typename T_F, typename T_T, typename A, size_t D>
struct converter_one<std::vector<T_F, A>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L& l, const std::vector<T_F, A>& from){
        debug_convert("convert_one");
        etl::dyn_matrix<T_T, D> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

// Convert a list<T_F> to a dyn_matrix<T_T, D>
template<typename T_F, typename T_T, typename A, size_t D>
struct converter_one<std::list<T_F, A>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L& l, const std::list<T_F, A>& from){
        debug_convert("convert_one");
        etl::dyn_matrix<T_T, D> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

// Convert a deque<T_F> to a dyn_matrix<T_T, D>
template<typename T_F, typename T_T, typename A, size_t D>
struct converter_one<std::deque<T_F, A>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L& l, const std::deque<T_F, A>& from){
        debug_convert("convert_one");
        etl::dyn_matrix<T_T, D> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

// Convert a vector<T_F> to a fast_dyn_matrix<T_T, Dims...>
template<typename T_F, typename T_T, typename A, size_t... Dims>
struct converter_one<std::vector<T_F, A>, etl::fast_dyn_matrix<T_T, Dims...>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims...> convert(const L&, const std::vector<T_F, A>& from){
        debug_convert("convert_one");
        return etl::fast_dyn_matrix<T_T, Dims...>(from);
    }
};

// Convert a list<T_F> to a fast_dyn_matrix<T_T, Dims...>
template<typename T_F, typename T_T, typename A, size_t... Dims>
struct converter_one<std::list<T_F, A>, etl::fast_dyn_matrix<T_T, Dims...>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims...> convert(const L&, const std::list<T_F, A>& from){
        debug_convert("convert_one");
        return etl::fast_dyn_matrix<T_T, Dims...>(from);
    }
};

// Convert a deque<T_F> to a fast_dyn_matrix<T_T, Dims...>
template<typename T_F, typename T_T, typename A, size_t... Dims>
struct converter_one<std::deque<T_F, A>, etl::fast_dyn_matrix<T_T, Dims...>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims...> convert(const L&, const std::deque<T_F, A>& from){
        debug_convert("convert_one");
        return etl::fast_dyn_matrix<T_T, Dims...>(from);
    }
};

// Convert a fast_dyn_matrix<T_F, Dims...> to a fast_dyn_matrix<T_T, Dims2...>
template<typename T_F, typename T_T, size_t... Dims, size_t... Dims2>
struct converter_one<etl::fast_dyn_matrix<T_F, Dims...>, etl::fast_dyn_matrix<T_T, Dims2...>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims2...> convert(const L&, const etl::fast_dyn_matrix<T_F, Dims...>& from){
        debug_convert("convert_one");
        return etl::fast_dyn_matrix<T_T, Dims2...>(from);
    }
};

// Convert a fast_matrix<T_F, Dims...> to a fast_dyn_matrix<T_T, Dims2...>
template<typename T_F, typename T_T, size_t... Dims, size_t... Dims2>
struct converter_one<etl::fast_matrix<T_F, Dims...>, etl::fast_dyn_matrix<T_T, Dims2...>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims2...> convert(const L&, const etl::fast_matrix<T_F, Dims...>& from){
        debug_convert("convert_one");
        return etl::fast_dyn_matrix<T_T, Dims2...>(from);
    }
};

// Convert a fast_dyn_matrix<T_F, Dims...> to a dyn_matrix<T_T, D>
template<typename T_F, typename T_T, size_t... Dims, size_t D>
struct converter_one<etl::fast_dyn_matrix<T_F, Dims...>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L& l, const etl::fast_dyn_matrix<T_F, Dims...>& from){
        debug_convert("convert_one");
        etl::dyn_matrix<T_T, D> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

// Convert a fast_matrix<T_F, Dims...> to a dyn_matrix<T_T, D>
template<typename T_F, typename T_T, size_t... Dims, size_t D>
struct converter_one<etl::fast_matrix<T_F, Dims...>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L& l, const etl::fast_matrix<T_F, Dims...>& from){
        debug_convert("convert_one");
        etl::dyn_matrix<T_T, D> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

// Convert a dyn_matrix<T_F, D> to a dyn_matrix<T_T, D>
template<typename T_F, typename T_T, size_t D>
struct converter_one<etl::dyn_matrix<T_F, D>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L&, const etl::dyn_matrix<T_F, D>& from){
        debug_convert("convert_one");
        return etl::dyn_matrix<T_T, D>(from);
    }
};

// Convert a dyn_matrix<T_F, D> to a fast_dyn_matrix<T_T, Dims...>
template<typename T_F, typename T_T, size_t D, size_t... Dims>
struct converter_one<etl::dyn_matrix<T_F, D>, etl::fast_dyn_matrix<T_T, Dims...>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims...> convert(const L&, const etl::dyn_matrix<T_F, D>& from){
        debug_convert("convert_one");
        return etl::fast_dyn_matrix<T_T, Dims...>(from);
    }
};

// Convert a dyn_matrix<T_F, D> to a dyn_matrix<T_T, D2>
template<typename T_F, typename T_T, size_t D, size_t D2>
struct converter_one<etl::dyn_matrix<T_F, D>, etl::dyn_matrix<T_T, D2>> {
    static_assert(std::is_convertible<T_F, T_T>::value, "DLL cannot convert your value type to the weight type (one)");

    template<typename L>
    static etl::dyn_matrix<T_T, D2> convert(const L& l, const etl::dyn_matrix<T_F, D>& from){
        debug_convert("convert_one");
        etl::dyn_matrix<T_T, D2> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

template<typename From, typename To>
struct converter_many {
    static_assert(cannot_convert<From, To>::value, "DLL does not know how to convert your input type (many)");
};

// Only convert the sub types not the outer container
template<template<typename...> class Container, typename From, typename To>
struct converter_many <Container<From>, Container<To>> {
    template<typename L>
    static Container<To> convert(const L& l, const Container<From>& from){
        debug_convert("convert_many_start");
        Container<To> to;
        to.reserve(from.size());
        for(auto& value : from){
            to.emplace_back(converter_one<From, To>::convert(l, value));
        }
        debug_convert("convert_many_end");
        return to;
    }
};

} //end of dll namespace
