//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "util/tmp.hpp"

namespace dll {

namespace detail {

/*!
 * \brief Helper traits indicate if the set contains dynamic layers
 */
template <typename... Layers>
constexpr const bool is_dynamic = (layer_traits<Layers>::is_dynamic() || ...);

/*!
 * \brief Helper traits indicate if the set contains convolutional layers
 */
template <typename... Layers>
constexpr const bool is_convolutional = (layer_traits<Layers>::is_convolutional_layer() || ...);

/*!
 * \brief Helper traits indicate if the set contains denoising layers
 */
template <typename... Layers>
constexpr const bool is_denoising = (layer_traits<Layers>::is_dense_rbm_layer() && ...);

/*!
 * \brief Indicates if the layer is a RBM shuffle layer
 */
template <typename Layer, typename Enable = void>
struct has_shuffle_helper;

/*!
 * \brief Indicates if the layer is a RBM shuffle layer
 */
template <typename Layer>
struct has_shuffle_helper <Layer, std::enable_if_t<layer_traits<Layer>::is_rbm_layer()>> {
    /*!
     * \brief true if the layer can be shuffled, false otherwise.
     */
    static constexpr bool value = rbm_layer_traits<Layer>::has_shuffle();
};

/*!
 * \brief Indicates if the layer is a RBM shuffle layer
 */
template <typename Layer>
struct has_shuffle_helper <Layer, std::enable_if_t<!layer_traits<Layer>::is_rbm_layer()>> {
    /*!
     * \brief true if the layer can be shuffled, false otherwise.
     */
    static constexpr bool value = false;
};

/*!
 * \brief Helper traits indicate if the set contains shuffle layers
 */
template <typename... Layers>
constexpr const bool has_shuffle_layer = (... || has_shuffle_helper<Layers>::value);

// TODO validate_layer_pair should be made more robust when
// transform layer are present between layers

template <typename L1, typename L2, typename Enable = void>
struct validate_layer_pair;

template <typename L1, typename L2>
struct validate_layer_pair<L1, L2, std::enable_if_t<layer_traits<L1>::is_transform_layer() || layer_traits<L2>::is_transform_layer()>>
    : std::true_type {};

template <typename L1, typename L2>
struct validate_layer_pair<L1, L2, cpp::disable_if_t<layer_traits<L1>::is_transform_layer() || layer_traits<L2>::is_transform_layer()>> : std::bool_constant<L1::output_size() == L2::input_size()> {};

template <typename... Layers>
struct validate_layers_impl;

template <typename Layer>
struct validate_layers_impl<Layer> : std::true_type {};

template <typename L1, typename L2, typename... Layers>
struct validate_layers_impl<L1, L2, Layers...> : std::bool_constant<
                                                     validate_layer_pair<L1, L2>::value &&
                                                     validate_layers_impl<L2, Layers...>::value> {};

//Note: It is not possible to add a template parameter with default value (SFINAE) on a variadic struct
//therefore, implementing the traits as function is nicer than nested structures

template <typename... Layers>
constexpr bool are_layers_valid() {
    if constexpr (!is_dynamic<Layers...>) {
        return validate_layers_impl<Layers...>();
    } else {
        return true;
    }
}

template <typename... Layers>
struct validate_label_layers;

template <typename Layer>
struct validate_label_layers<Layer> : std::true_type {};

template <typename L1, typename L2, typename... Layers>
struct validate_label_layers<L1, L2, Layers...> : std::bool_constant<
                                                      L1::output_size() <= L2::input_size() &&
                                                      validate_label_layers<L2, Layers...>::value> {};

} // end of namespace detail

namespace detail {

/*!
 * \brief A leaf in the list of layers.
 */
template <size_t I, typename T>
struct layers_leaf {
    T value; ///< The value of the leaf

    /*!
     * \brief Returns a reference to the value of the layer
     */
    T& get() noexcept {
        return value;
    }

    /*!
     * \brief Returns a const reference to the value of the layer
     */
    const T& get() const noexcept {
        return value;
    }
};

template <typename Indices, typename... Layers>
struct layers_impl;

template <size_t... I, typename... Layers>
struct layers_impl<std::index_sequence<I...>, Layers...> : layers_leaf<I, Layers>... {};

/*!
 * \brief The layers of a DBN
 */
template <bool Labels, typename... Layers>
struct layers;

/*!
 * \brief The layers of a DBN
 */
template <typename... Layers>
struct layers <false, Layers...> {
    static constexpr size_t size = sizeof...(Layers); ///< The number of layers in the set

    static constexpr bool is_dynamic        = detail::is_dynamic<Layers...>;        ///< Indicates if the set contains dynamic layers
    static constexpr bool is_convolutional  = detail::is_convolutional<Layers...>;  ///< Indicates if the set contains convolutional layers
    static constexpr bool is_denoising      = detail::is_denoising<Layers...>;      ///< Indicates if the set contains denoising layers
    static constexpr bool has_shuffle_layer = detail::has_shuffle_layer<Layers...>; ///< Indicates if the set contains shuffle layers

    static_assert(size > 0, "A network must have at least 1 layer");
    static_assert(detail::are_layers_valid<Layers...>(), "The inner sizes of the layers must correspond");

    using base_t      = layers_impl<std::make_index_sequence<size>, Layers...>;
    using layers_list = cpp::type_list<Layers...>;

    base_t base;
};

/*!
 * \brief The layers of a DBN
 */
template <typename... Layers>
struct layers <true, Layers...> {
    static constexpr size_t size = sizeof...(Layers); ///< The number of layers in the set

    static constexpr bool is_dynamic        = false;                                  ///< Indicates if the set contains dynamic layers
    static constexpr bool is_convolutional  = false;                                  ///< Indicates if the set contains convolutional layers
    static constexpr bool is_denoising      = false;                                  ///< Indicates if the set contains denoising layers
    static constexpr bool has_shuffle_layer = detail::has_shuffle_layer<Layers...>; ///< Indicates if the set contains shuffle layers

    static_assert(size > 0, "A network must have at least 1 layer");
    static_assert(detail::validate_label_layers<Layers...>::value, "The inner sizes of RBM must correspond");
    static_assert(!detail::is_dynamic<Layers...>, "dbn_label_layers should not be used with dynamic RBMs");

    using base_t      = layers_impl<std::make_index_sequence<size>, Layers...>; ///< The base implementation for layers
    using layers_list = cpp::type_list<Layers...>;                              ///< The list of layers

    base_t base; ///< The tuple structure to hold all layers
};

//Note: Maybe simplify further removing the type_list

/*!
 * \brief Get the type of a layer by index
 * \tparam I The index of the layer
 * \tparam T The set of layers
 */
template <size_t I, typename T>
struct layer_type;

/*!
 * \copydoc layer_type
 */
template <size_t I>
struct layer_type<I, cpp::type_list<>> {
    static_assert(I == 0, "index out of range");
    static_assert(I != 0, "index out of range");
};

/*!
 * \copydoc layer_type
 */
template <typename Head, typename... T>
struct layer_type<0, cpp::type_list<Head, T...>> {
    /*!
     * \brief The type of the layer
     */
    using type = Head;
};

/*!
 * \copydoc layer_type
 */
template <size_t I, typename Head, typename... T>
struct layer_type<I, cpp::type_list<Head, T...>> {
    /*!
     * \brief The type of the layer
     */
    using type = typename layer_type<I - 1, cpp::type_list<T...>>::type;
};

/*!
 * \copydoc layer_type
 */
template <size_t I, bool Labels, typename... Layers>
struct layer_type<I, layers<Labels, Layers...>> {
    /*!
     * \brief The type of the layer
     */
    using type = typename layer_type<I, cpp::type_list<Layers...>>::type;
};

/*!
 * \brief Get the type of a layer by index
 * \tparam I The index of the layer
 * \tparam Layers The set of layers
 */
template <size_t I, typename Layers>
using layer_type_t = typename layer_type<I, Layers>::type;

/*!
 * \brief Return the Ith layer in the given layer set
 * \param layers The layers holder
 * \tparam I The layer to get
 * \return a reference to the Ith layer in the given layer set
 */
template <size_t I, typename Layers>
layer_type_t<I, Layers>& layer_get(Layers& layers) {
    return static_cast<layers_leaf<I, layer_type_t<I, Layers>>&>(layers.base).get();
}

/*!
 * \brief Return the Ith layer in the given layer set
 * \param layers The layers holder
 * \tparam I The layer to get
 * \return a reference to the Ith layer in the given layer set
 */
template <size_t I, typename Layers>
const layer_type_t<I, Layers>& layer_get(const Layers& layers) {
    return static_cast<const layers_leaf<I, layer_type_t<I, Layers>>&>(layers.base).get();
}

template <typename DBN, typename Functor, size_t... I>
void for_each_layer_type_sub(Functor&& functor, const std::index_sequence<I...>& /* i */) {
    (functor(static_cast<typename DBN::template layer_type<I>*>(nullptr)), ...);
}

template <typename DBN, typename Functor>
void for_each_layer_type(Functor&& functor) {
    for_each_layer_type_sub<DBN>(functor, std::make_index_sequence<DBN::layers>());
}

} //end of namespace detail

/*!
 * \brief Holder for the layers of a DBN
 */
template <typename... Layers>
using dbn_layers = detail::layers<false, Layers...>;

/*!
 * \brief Holder for the layers of a network
 */
template <typename... Layers>
using network_layers = detail::layers<false, Layers...>;

/*!
 * \brief Holder for the layers of a DBN, training with labels + RBM in last layer
 */
template <typename... Layers>
using dbn_label_layers = detail::layers<true, Layers...>;

} //end of namespace dll
