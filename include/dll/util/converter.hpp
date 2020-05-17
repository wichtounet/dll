//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

#define debug_convert(X) etl::inc_counter(X)

/*!
 * \brief Simple helper to form nicer display for static_assert
 */
template<typename From, typename To>
inline constexpr bool value = false;

/*!
 * \brief Converter utility to converty from type *From* to type *To*.
 */
template<typename From, typename To, typename Enable = void>
struct converter_one {
    static_assert(cannot_convert<From, To>, "DLL does not know how to convert your input type (one)");
};

// No conversion
// This should only happen when used from converter_many

/*!
 * \copydoc converter_one
 */
template<typename From>
struct converter_one <From, From> {
    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static const From& convert(const L&, const From& from){
        return from;
    }
};

// Convert a vector<T_F> to a dyn_vector<T_T>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, typename A>
struct converter_one<std::vector<T_F, A>, etl::dyn_matrix<T_T, 1>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::dyn_matrix<T_T, 1> convert(const L&, const std::vector<T_F, A>& from){
        debug_convert("converter::one");
        etl::dyn_matrix<T_T, 1> c;
        c = from;
        return c;
    }
};

// Convert a list<T_F> to a dyn_vector<T_T>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, typename A>
struct converter_one<std::list<T_F, A>, etl::dyn_matrix<T_T, 1>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::dyn_matrix<T_T, 1> convert(const L&, const std::list<T_F, A>& from){
        debug_convert("converter::one");
        etl::dyn_matrix<T_T, 1> c;
        c = from;
        return c;
    }
};

// Convert a deque<T_F> to a dyn_vector<T_T>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, typename A>
struct converter_one<std::deque<T_F, A>, etl::dyn_matrix<T_T, 1>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::dyn_matrix<T_T, 1> convert(const L&, const std::deque<T_F, A>& from){
        debug_convert("converter::one");
        etl::dyn_matrix<T_T, 1> c;
        c = from;
        return c;
    }
};

// Convert a vector<T_F> to a dyn_matrix<T_T, D>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, typename A, size_t D>
struct converter_one<std::vector<T_F, A>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L& l, const std::vector<T_F, A>& from){
        debug_convert("converter::one");
        etl::dyn_matrix<T_T, D> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

// Convert a list<T_F> to a dyn_matrix<T_T, D>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, typename A, size_t D>
struct converter_one<std::list<T_F, A>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L& l, const std::list<T_F, A>& from){
        debug_convert("converter::one");
        etl::dyn_matrix<T_T, D> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

// Convert a deque<T_F> to a dyn_matrix<T_T, D>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, typename A, size_t D>
struct converter_one<std::deque<T_F, A>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L& l, const std::deque<T_F, A>& from){
        debug_convert("converter::one");
        etl::dyn_matrix<T_T, D> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

// Convert a vector<T_F> to a fast_dyn_matrix<T_T, Dims...>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, typename A, size_t... Dims>
struct converter_one<std::vector<T_F, A>, etl::fast_dyn_matrix<T_T, Dims...>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims...> convert(const L&, const std::vector<T_F, A>& from){
        debug_convert("converter::one");
        etl::fast_dyn_matrix<T_T, Dims...> c;
        c = from;
        return c;
    }
};

// Convert a list<T_F> to a fast_dyn_matrix<T_T, Dims...>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, typename A, size_t... Dims>
struct converter_one<std::list<T_F, A>, etl::fast_dyn_matrix<T_T, Dims...>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims...> convert(const L&, const std::list<T_F, A>& from){
        debug_convert("converter::one");
        etl::fast_dyn_matrix<T_T, Dims...> c;
        c = from;
        return c;
    }
};

// Convert a deque<T_F> to a fast_dyn_matrix<T_T, Dims...>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, typename A, size_t... Dims>
struct converter_one<std::deque<T_F, A>, etl::fast_dyn_matrix<T_T, Dims...>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims...> convert(const L&, const std::deque<T_F, A>& from){
        debug_convert("converter::one");
        etl::fast_dyn_matrix<T_T, Dims...> c;
        c = from;
        return c;
    }
};

// Convert a fast_dyn_matrix<T_F, Dims...> to a fast_dyn_matrix<T_T, Dims2...>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, size_t... Dims, size_t... Dims2>
struct converter_one<etl::fast_dyn_matrix<T_F, Dims...>, etl::fast_dyn_matrix<T_T, Dims2...>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims2...> convert(const L&, const etl::fast_dyn_matrix<T_F, Dims...>& from){
        debug_convert("converter::one");
        etl::fast_dyn_matrix<T_T, Dims2...> c;
        c = from;
        return c;
    }
};

// Convert a fast_matrix<T_F, Dims...> to a fast_dyn_matrix<T_T, Dims2...>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, size_t... Dims, size_t... Dims2>
struct converter_one<etl::fast_matrix<T_F, Dims...>, etl::fast_dyn_matrix<T_T, Dims2...>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims2...> convert(const L&, const etl::fast_matrix<T_F, Dims...>& from){
        debug_convert("converter::one");
        etl::fast_dyn_matrix<T_T, Dims2...> c;
        c = from;
        return c;
    }
};

// Convert a fast_dyn_matrix<T_F, Dims...> to a dyn_matrix<T_T, D>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, size_t... Dims, size_t D>
struct converter_one<etl::fast_dyn_matrix<T_F, Dims...>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L& l, const etl::fast_dyn_matrix<T_F, Dims...>& from){
        debug_convert("converter::one");
        etl::dyn_matrix<T_T, D> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

// Convert a fast_matrix<T_F, Dims...> to a dyn_matrix<T_T, D>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, size_t... Dims, size_t D>
struct converter_one<etl::fast_matrix<T_F, Dims...>, etl::dyn_matrix<T_T, D>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L& l, const etl::fast_matrix<T_F, Dims...>& from){
        debug_convert("converter::one");
        etl::dyn_matrix<T_T, D> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

// Convert a dyn_matrix<T_F, D> to a fast_dyn_matrix<T_T, Dims...>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, size_t D, size_t... Dims>
struct converter_one<etl::dyn_matrix<T_F, D>, etl::fast_dyn_matrix<T_T, Dims...>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::fast_dyn_matrix<T_T, Dims...> convert(const L&, const etl::dyn_matrix<T_F, D>& from){
        debug_convert("converter::one");
        etl::fast_dyn_matrix<T_T, Dims...> c;
        c = from;
        return c;
    }
};

// Convert a dyn_matrix<T_F, D> to a dyn_matrix<T_T, D>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, size_t D>
struct converter_one<etl::dyn_matrix<T_F, D>, etl::dyn_matrix<T_T, D>, std::enable_if_t<!std::is_same_v<T_F, T_T>>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::dyn_matrix<T_T, D> convert(const L&, const etl::dyn_matrix<T_F, D>& from){
        debug_convert("converter::one");
        etl::dyn_matrix<T_T, D> c;
        c = from;
        return c;
    }
};

// Convert a dyn_matrix<T_F, D> to a dyn_matrix<T_T, D2>

/*!
 * \copydoc converter_one
 */
template<typename T_F, typename T_T, size_t D, size_t D2>
struct converter_one<etl::dyn_matrix<T_F, D>, etl::dyn_matrix<T_T, D2>, std::enable_if_t<(D != D2)>> {
    static_assert(std::is_convertible_v<T_F, T_T>, "DLL cannot convert your value type to the weight type (one)");

    /*!
     * \brief Convert from the given container into the specific type
     * \param l The layer for which to convert
     * \param from The container to convert from
     * \return the converted result
     */
    template<typename L>
    static etl::dyn_matrix<T_T, D2> convert(const L& l, const etl::dyn_matrix<T_F, D>& from){
        debug_convert("converter::one");
        etl::dyn_matrix<T_T, D2> converted;
        l.prepare_input(converted);
        converted = from;
        return converted;
    }
};

template<typename From, typename To>
struct converter_many {
    static_assert(cannot_convert<From, To>, "DLL does not know how to convert your input type (many)");
};

// Only convert the sub types not the outer container
template<template<typename...> typename Container, typename From, typename To>
struct converter_many <Container<From>, Container<To>> {
    template<typename L>
    static Container<To> convert(const L& l, const Container<From>& from){
        debug_convert("converter::many");
        Container<To> to;
        to.reserve(from.size());
        for(auto& value : from){
            to.emplace_back(converter_one<From, To>::convert(l, value));
        }
        return to;
    }
};

} //end of dll namespace
