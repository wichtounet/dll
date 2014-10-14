//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_LAYERS_HPP
#define DLL_DBN_LAYERS_HPP

#include "tmp.hpp"

namespace dll {

template<typename... Layers>
struct validate_layers;

template<typename Layer>
struct validate_layers <Layer> : std::true_type {};

template<typename L1, typename L2, typename... Layers>
struct validate_layers <L1, L2, Layers...> :
    std::integral_constant<bool,
        cpp::and_u<
            rbm_traits<L1>::output_size() == rbm_traits<L2>::input_size(),
            validate_layers<L2, Layers...>::value
        >::value> {};

template<typename... Layers>
struct validate_label_layers;

template<typename Layer>
struct validate_label_layers <Layer> : std::true_type {};

template<typename L1, typename L2, typename... Layers>
struct validate_label_layers <L1, L2, Layers...> :
    std::integral_constant<bool,
        cpp::and_u<
            rbm_traits<L1>::output_size() <= rbm_traits<L2>::input_size(),
            validate_label_layers<L2, Layers...>::value
        >::value> {};

/**
 * \brief Simple placeholder for a collection of layers
 */
template<typename... Layers>
struct dbn_layers {
    static constexpr const std::size_t layers = sizeof...(Layers);

    static_assert(layers > 0, "A DBN must have at least 1 layer");
    static_assert(validate_layers<Layers...>::value, "The inner sizes of RBM must correspond");

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

    static_assert(layers > 0, "A DBN must have at least 1 layer");
    static_assert(validate_label_layers<Layers...>::value, "The inner sizes of RBM must correspond");

    using tuple_type = std::tuple<Layers...>;
};

} //end of namespace dll

#endif
