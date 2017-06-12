//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

// extract_weight

/*!
 * \brief Extrac the weight type from the set of the layers of the DBN
 */
template <size_t I, typename DBN, typename Enable = void>
struct extract_weight_t;

/*!
 * \copydoc extract_weight_t
 */
template <size_t I, typename DBN>
struct extract_weight_t<I, DBN, std::enable_if_t<layer_traits<typename DBN::template layer_type<I>>::has_same_type()>> {
    /*!
     * \brief The extracted weight type
     */
    using type = typename extract_weight_t<I + 1, DBN>::type;
};

/*!
 * \copydoc extract_weight_t
 */
template <size_t I, typename DBN>
struct extract_weight_t<I, DBN, cpp::disable_if_t<layer_traits<typename DBN::template layer_type<I>>::has_same_type()>> {
    /*!
     * \brief The extracted weight type
     */
    using type = typename DBN::template layer_type<I>::weight;
};

// test that a layer has the correct weight type

template <size_t I, typename DBN, typename T, typename Enable = void>
struct weight_type_same;

template <size_t I, typename DBN, typename T>
struct weight_type_same<I, DBN, T, std::enable_if_t<layer_traits<typename DBN::template layer_type<I>>::has_same_type()>> {
    static constexpr bool value = true;
};

template <size_t I, typename DBN, typename T>
struct weight_type_same<I, DBN, T, cpp::disable_if_t<layer_traits<typename DBN::template layer_type<I>>::has_same_type()>> {
    using type = typename DBN::template layer_type<I>::weight;
    static constexpr bool value = std::is_same<T, type>::value;
};

// validate the weight type of the layers

template <size_t I, typename DBN, typename T, typename Enable = void>
struct validate_weight_type_impl;

template <size_t I, typename DBN, typename T>
struct validate_weight_type_impl< I, DBN, T, std::enable_if_t<(I == DBN::layers_t::size - 1)> > {
    static constexpr bool value = weight_type_same<I, DBN, T>::value;
};

template <size_t I, typename DBN, typename T>
struct validate_weight_type_impl< I, DBN, T, std::enable_if_t<(I < DBN::layers_t::size - 1)> > {
    static constexpr bool value = weight_type_same<I, DBN, T>::value && validate_weight_type_impl<I + 1, DBN, T>::value;
};

template <typename DBN, typename T>
struct validate_weight_type {
    static constexpr bool value = validate_weight_type_impl<0, DBN, T>::value;
};

// Compute the distance between two iterators, only if random_access

template <typename Iterator>
size_t fast_distance(Iterator& first, Iterator& last) {
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
void safe_advance(Iterator& it, const Iterator& end, size_t distance) {
    size_t i = 0;
    while (it != end && i < distance) {
        ++it;
        ++i;
    }
}

template <typename D, size_t N, typename T>
struct for_each_impl;

template <typename D, size_t... I>
struct for_each_impl<D, 1, std::index_sequence<I...>> {
    D& dbn;

    for_each_impl(D& dbn)
            : dbn(dbn) {}

    template <typename Functor>
    void for_each_layer(Functor&& functor) {
        functor(dbn.template layer_get<0>());
    }

    template <typename Functor>
    void for_each_layer_i(Functor&& functor) {
        functor(dbn.template layer_get<0>(), 0);
    }

    template <typename Functor>
    void for_each_layer_pair(Functor&& functor) {
        cpp_unused(functor);
        // Nothing to do here
    }

    template <typename Functor>
    void for_each_layer_pair_i(Functor&& functor) {
        cpp_unused(functor);
        // Nothing to do here
    }

    template <typename Functor>
    void for_each_layer_rpair(Functor&& functor) {
        cpp_unused(functor);
        // Nothing to do here
    }

    template <typename Functor>
    void for_each_layer_rpair_i(Functor&& functor) {
        cpp_unused(functor);
        // Nothing to do here
    }
};

template <typename D, size_t N, size_t... I>
struct for_each_impl<D, N, std::index_sequence<I...>> {
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
