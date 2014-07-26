//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_LAYERS_HPP
#define DLL_DBN_LAYERS_HPP

namespace dll {

namespace detail {

template<typename... R>
struct check_rbm ;

template<typename R1, typename... R>
struct check_rbm<R1, R...> : std::integral_constant<bool, and_u<rbm_traits<R1>::in_dbn(), check_rbm<R...>::value>::value> {};

template<typename R1>
struct check_rbm<R1> : std::integral_constant<bool, R1::DBN> {};

} //end of namespace detail

/**
 * \brief Simple placeholder for a collection of layers
 */
template<typename... Layers>
struct dbn_layers {
    static constexpr const std::size_t layers = sizeof...(Layers);

    static_assert(layers > 0, "A DBN must have at least 1 layer");
    static_assert(detail::check_rbm<Layers...>::value, "RBM must be in DBN mode");

    using tuple_type = std::tuple<Layers...>;
};

} //end of namespace dll

#endif
