//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_LAYERS_HPP
#define DLL_DBN_LAYERS_HPP

#include "tmp.hpp"

namespace dll {

namespace detail {

template<typename... Layers>
struct is_dynamic : cpp::or_u<layer_traits<Layers>::is_dynamic()...> {};

template<typename... Layers>
struct is_convolutional : cpp::or_u<layer_traits<Layers>::is_convolutional()...> {};

template<typename... Layers>
struct is_multiplex : cpp::or_u<layer_traits<Layers>::is_multiplex_layer()...> {};

// TODO validate_layer_pair should be made more robust when
// transform layer are present between layers

template<typename L1, typename L2, typename Enable = void>
struct validate_layer_pair;

template<typename L1, typename L2>
struct validate_layer_pair <L1, L2, std::enable_if_t<layer_traits<L1>::is_transform_layer() || layer_traits<L2>::is_transform_layer()>>
    : std::true_type {};

template<typename L1, typename L2>
struct validate_layer_pair <L1, L2, cpp::disable_if_t<layer_traits<L1>::is_transform_layer() || layer_traits<L2>::is_transform_layer()>> : cpp::bool_constant<layer_traits<L1>::output_size() == layer_traits<L2>::input_size()> {};

template<typename... Layers>
struct validate_layers_impl;

template<typename Layer>
struct validate_layers_impl <Layer> : std::true_type {};

template<typename L1, typename L2, typename... Layers>
struct validate_layers_impl <L1, L2, Layers...> :
    cpp::bool_constant_c<
        cpp::and_u<
            validate_layer_pair<L1, L2>::value,
            validate_layers_impl<L2, Layers...>::value
        >> {};

//Note: It is not possible to add a template parameter with default value (SFINAE) on a variadic struct
//therefore, implementing the traits as function is nicer than nested structures

template<typename... Layers, cpp_enable_if(is_dynamic<Layers...>::value)>
constexpr bool are_layers_valid(){
    return true;
}

template<typename... Layers, cpp_disable_if(is_dynamic<Layers...>::value)>
constexpr bool are_layers_valid(){
    return validate_layers_impl<Layers...>();
}

template<typename... Layers>
struct validate_label_layers;

template<typename Layer>
struct validate_label_layers <Layer> : std::true_type {};

template<typename L1, typename L2, typename... Layers>
struct validate_label_layers <L1, L2, Layers...> :
    cpp::bool_constant_c<
        cpp::and_u<
            layer_traits<L1>::output_size() <= layer_traits<L2>::input_size(),
            validate_label_layers<L2, Layers...>::value
        >> {};

} // end of namespace detail

/**
 * \brief Simple placeholder for a collection of layers
 */
template<typename... Layers>
struct dbn_layers {
    static constexpr const std::size_t layers = sizeof...(Layers);
    static constexpr const bool is_dynamic = detail::is_dynamic<Layers...>();
    static constexpr const bool is_convolutional = detail::is_convolutional<Layers...>();
    static constexpr const bool is_multiplex = detail::is_multiplex<Layers...>();

    static_assert(layers > 0, "A DBN must have at least 1 layer");
    static_assert(detail::are_layers_valid<Layers...>(), "The inner sizes of RBM must correspond");

    using tuple_type = std::tuple<Layers...>;
};

/**
 * \brief Simple placeholder for a collection of layers
 *
 * This version has to be used instead of dbn_layers when labels are placed in
 * the last layer.
 */
template<typename... Layers>
struct dbn_label_layers {
    static constexpr const std::size_t layers = sizeof...(Layers);
    static constexpr const bool is_dynamic = false;
    static constexpr const bool is_convolutional = false;   //There is no support for convolutional RBM and labels
    static constexpr const bool is_multiplex = false;       //There is no support for multiplex layers and labels

    static_assert(layers > 0, "A DBN must have at least 1 layer");
    static_assert(detail::validate_label_layers<Layers...>::value, "The inner sizes of RBM must correspond");
    static_assert(!detail::is_dynamic<Layers...>(), "dbn_label_layers should not be used with dynamic RBMs");

    using tuple_type = std::tuple<Layers...>;
};

} //end of namespace dll

#endif
