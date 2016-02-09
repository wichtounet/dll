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

#ifndef DLL_DETAILS_HPP
#define DLL_DETAILS_HPP

namespace dll {

namespace dbn_detail {

//TODO Could be good to ensure that either a) all layer have the same weight b) use the correct type for each layer

template <std::size_t I, typename DBN, typename Enable = void>
struct extract_weight_t;

template <std::size_t I, typename DBN>
struct extract_weight_t<I, DBN, std::enable_if_t<layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = typename extract_weight_t<I + 1, DBN>::type;
};

template <std::size_t I, typename DBN>
struct extract_weight_t<I, DBN, cpp::disable_if_t<layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
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

template <typename Iterator>
void safe_advance(Iterator& it, const Iterator& end, std::size_t distance) {
    std::size_t i = 0;
    while (it != end && i < distance) {
        ++it;
        ++i;
    }
}

template <typename DBN, std::size_t I, typename Enable = void>
struct layer_input_simple;

template <typename DBN, std::size_t I>
struct layer_input_simple<DBN, I, std::enable_if_t<!layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = typename DBN::template layer_type<I>::input_one_t;
};

template <typename DBN, std::size_t I>
struct layer_input_simple<DBN, I, std::enable_if_t<layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = typename layer_input_simple<DBN, I + 1>::type;
};

template <typename DBN, std::size_t I, typename Enable = void>
struct layer_input_batch;

template <typename DBN, std::size_t I>
struct layer_input_batch<DBN, I, std::enable_if_t<!layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    template <std::size_t B>
    using type = typename DBN::template layer_type<I>::template input_batch_t<B>;
};

template <typename DBN, std::size_t I>
struct layer_input_batch<DBN, I, std::enable_if_t<layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    template <std::size_t B>
    using type = typename layer_input_batch<DBN, I + 1>::template type<B>;
};

// Get the output type for one sample of the given layer
template <typename DBN, std::size_t I, typename Enable = void>
struct layer_output_one;

// Get the output type for one sample of the given layer
template <typename DBN, std::size_t I>
using layer_output_one_t = typename layer_output_one<DBN, I>::type;

// Get the output type for multiple sample of the given layer
template <typename DBN, std::size_t I, typename Enable = void>
struct layer_output;

// Get the output type for multiple sample of the given layer
template <typename DBN, std::size_t I>
using layer_output_t = typename layer_output<DBN, I>::type;

// Get the input type for one sample of the given layer
template <typename DBN, std::size_t I, typename Enable = void>
struct layer_input_one;

// Get the input type for one sample of the given layer
template <typename DBN, std::size_t I>
using layer_input_one_t = typename layer_input_one<DBN, I>::type;

// Get the input type for multiple sample of the given layer
template <typename DBN, std::size_t I, typename Enable = void>
struct layer_input;

// Get the input type for multiple sample of the given layer
template <typename DBN, std::size_t I>
using layer_input_t = typename layer_input<DBN, I>::type;

//A standard layer has its own output type
template <typename DBN, std::size_t I>
struct layer_output_one<DBN, I, std::enable_if_t<!layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = typename DBN::template layer_type<I>::output_one_t;
};

//A transform layer don't change the type
template <typename DBN, std::size_t I>
struct layer_output_one<DBN, I, std::enable_if_t<layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = std::conditional_t<I == 0, typename DBN::input_t, layer_input_one_t<DBN, I>>;
};

//A standard layer has its own output type
template <typename DBN, std::size_t I>
struct layer_output<DBN, I, std::enable_if_t<!layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = typename DBN::template layer_type<I>::output_t;
};

//A transform layer don't change the type
template <typename DBN, std::size_t I>
struct layer_output<DBN, I, std::enable_if_t<layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    //TODO This one needs to be checked by a test case
    using type = std::conditional_t<I == 0, typename DBN::input_t, layer_input_t<DBN, I>>;
};

//A standard type has its own input type
template <typename DBN, std::size_t I>
struct layer_input_one<DBN, I, std::enable_if_t<!layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = typename DBN::template layer_type<I>::input_one_t;
};

//The first transform layer uses the input type of the next layer as input type
template <typename DBN, std::size_t I>
struct layer_input_one<DBN, I, std::enable_if_t<I == 0 && layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = layer_input_one_t<DBN, I + 1>;
};

//A transform layer uses the output type of the previous layer
template <typename DBN, std::size_t I>
struct layer_input_one<DBN, I, std::enable_if_t<(I > 0) && layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = layer_output_one_t<DBN, I - 1>;
};

//A standard type has its own input type
template <typename DBN, std::size_t I>
struct layer_input<DBN, I, std::enable_if_t<!layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = typename DBN::template layer_type<I>::input_t;
};

//The first transform layer uses the input type of the next layer as input type
template <typename DBN, std::size_t I>
struct layer_input<DBN, I, std::enable_if_t<I == 0 && layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = layer_input_t<DBN, I + 1>;
};

//A transform layer uses the output type of the previous layer
template <typename DBN, std::size_t I>
struct layer_input<DBN, I, std::enable_if_t<(I > 0) && layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = layer_output_t<DBN, I - 1>;
};

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

#endif
