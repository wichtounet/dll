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
struct is_convolutional : cpp::or_u<layer_traits<Layers>::is_convolutional_rbm_layer()...> {};

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

namespace detail {

template<std::size_t I, typename T>
struct layers_leaf {
    T value;

    T& get() noexcept {
        return value;
    }

    const T& get() const noexcept {
        return value;
    }
};

template<typename Indices, typename... Layers>
struct layers_impl;

template<std::size_t... I, typename... Layers>
struct layers_impl <std::index_sequence<I...>, Layers...> : layers_leaf<I, Layers>... {

};

//Maybe make specialization for Labels = true, Labels = false

template<bool Labels, typename... Layers>
struct layers {
    static constexpr const std::size_t size = sizeof...(Layers);

    static constexpr const bool is_dynamic = Labels ? false : detail::is_dynamic<Layers...>();
    static constexpr const bool is_convolutional = Labels ? false : detail::is_convolutional<Layers...>();
    static constexpr const bool is_multiplex = Labels ? false : detail::is_multiplex<Layers...>();

    static_assert(size > 0, "A network must have at least 1 layer");
    //TODO static_assert(Labels ? detail::validate_label_layers<Layers...>::value : detail::are_layers_valid<Layers...>(), "The inner sizes of RBM must correspond");
    static_assert(!Labels || !detail::is_dynamic<Layers...>(), "dbn_label_layers should not be used with dynamic RBMs");

    using base_t = layers_impl<std::make_index_sequence<size>, Layers...>;
    using layers_list = cpp::type_list<Layers...>;

    base_t base;
};

//Note: Maybe simplify further removing the type_list

template<std::size_t I, typename T>
struct layer_type;

template<std::size_t I>
struct layer_type<I, cpp::type_list<> >{
    static_assert(I == 0, "index out of range");
    static_assert(I != 0, "index out of range");
};

template<typename Head, typename... T>
struct layer_type<0, cpp::type_list<Head, T...>> {
    using type = Head;
};

template<std::size_t I, typename Head, typename ... T>
struct layer_type<I, cpp::type_list<Head,  T...>> {
    using type = typename layer_type<I-1, cpp::type_list<T...> >::type;
};

template<std::size_t I, bool Labels, typename... Layers>
struct layer_type<I, layers<Labels, Layers...>> {
    using type = typename layer_type<I, cpp::type_list<Layers...> >::type;
};

template<std::size_t I, typename Layers>
using layer_type_t = typename layer_type<I, Layers>::type;

template<std::size_t I, typename Layers>
layer_type_t<I, Layers>& layer_get(Layers& layers){
    return static_cast<layers_leaf<I, layer_type_t<I, Layers>>&>(layers.base).get();
}

template<std::size_t I, typename Layers>
const layer_type_t<I, Layers>& layer_get(const Layers& layers){
    return static_cast<const layers_leaf<I, layer_type_t<I, Layers>>&>(layers.base).get();
}

template<typename DBN, typename Functor, std::size_t... I>
void for_each_layer_type_sub(Functor&& functor, const std::index_sequence<I...>& /* i */){
    int wormhole[] = {(functor(static_cast<typename DBN::template layer_type<I>*>(nullptr)),0)...};
    cpp_unused(wormhole);
}

template<typename DBN, typename Functor>
void for_each_layer_type(Functor&& functor){
    for_each_layer_type_sub<DBN>(functor, std::make_index_sequence<DBN::layers>());
}

} //end of namespace detail

template<typename... Layers>
using dbn_layers = detail::layers<false, Layers...>;

template<typename... Layers>
using dbn_label_layers = detail::layers<false, Layers...>;

} //end of namespace dll

#endif
