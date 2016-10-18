//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file dbn_detail.hpp
 * \brief DBN implementation details
 */

#pragma once

namespace dll {

namespace dbn_detail {

//TODO Could be good to ensure that either a) all layer have the same weight b) use the correct type for each layer

template <std::size_t I, typename DBN, typename Enable = void>
struct extract_weight_t;

template <std::size_t I, typename DBN>
struct extract_weight_t<I, DBN, std::enable_if_t<layer_traits<typename DBN::template layer_type<I>>::has_same_type()>> {
    using type = typename extract_weight_t<I + 1, DBN>::type;
};

template <std::size_t I, typename DBN>
struct extract_weight_t<I, DBN, cpp::disable_if_t<layer_traits<typename DBN::template layer_type<I>>::has_same_type()>> {
    using type = typename DBN::template layer_type<I>::weight;
};

template <typename Iterator>
std::size_t fast_distance(Iterator& first, Iterator& last) {
    if (std::is_same<typename std::iterator_traits<Iterator>::iterator_category, std::random_access_iterator_tag>::value) {
        return std::distance(first, last);
    } else {
        return 0;
    }
}

template <typename Iterator, cpp_enable_if(std::is_same<typename std::iterator_traits<Iterator>::iterator_category, std::random_access_iterator_tag>::value)>
void safe_sort(Iterator first, Iterator last) {
    std::sort(first, last);
}

template <typename Iterator, cpp_disable_if(std::is_same<typename std::iterator_traits<Iterator>::iterator_category, std::random_access_iterator_tag>::value)>
void safe_sort(Iterator first, Iterator last) {
    cpp_unused(first);
    cpp_unused(last);
    //Nothing
}

template <typename Iterator>
void safe_advance(Iterator& it, const Iterator& end, std::size_t distance) {
    std::size_t i = 0;
    while (it != end && i < distance) {
        ++it;
        ++i;
    }
}

template <typename D, typename T>
struct for_each_impl;

template <typename D, std::size_t... I>
struct for_each_impl<D, std::index_sequence<I...>> {
    D& dbn;

    for_each_impl(D& dbn)
            : dbn(dbn) {}

    template <typename Functor>
    void for_each_layer(Functor&& functor) {
        int wormhole[] = {(functor(dbn.template layer_get<I>()), 0)...};
        cpp_unused(wormhole);
    }

    template <typename Functor>
    void for_each_layer_i(Functor&& functor) {
        int wormhole[] = {(functor(I, dbn.template layer_get<I>()), 0)...};
        cpp_unused(wormhole);
    }

    template <typename Functor>
    void for_each_layer_pair(Functor&& functor) {
        int wormhole[] = {(functor(dbn.template layer_get<I>(), dbn.template layer_get<I + 1>()), 0)...};
        cpp_unused(wormhole);
    }

    template <typename Functor>
    void for_each_layer_pair_i(Functor&& functor) {
        int wormhole[] = {(functor(I, dbn.template layer_get<I>(), dbn.template layer_get<I + 1>()), 0)...};
        cpp_unused(wormhole);
    }

    template <typename Functor>
    void for_each_layer_rpair(Functor&& functor) {
        int wormhole[] = {(functor(dbn.template layer_get<D::layers - I - 2>(), dbn.template layer_get<D::layers - I - 1>()), 0)...};
        cpp_unused(wormhole);
    }

    template <typename Functor>
    void for_each_layer_rpair_i(Functor&& functor) {
        int wormhole[] = {(functor(D::layers - I - 2, dbn.template layer_get<D::layers - I - 2>(), dbn.template layer_get<D::layers - I - 1>()), 0)...};
        cpp_unused(wormhole);
    }
};

} //end of namespace dbn_detail

} //end of namespace dll
