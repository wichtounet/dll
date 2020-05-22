//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "transform_layer.hpp"
#include "lcn.hpp"

namespace dll {

/*!
 * \brief Local Contrast Normalization layer
 */
template <typename Desc>
struct dyn_lcn_layer_impl : transform_layer<dyn_lcn_layer_impl<Desc>> {
    using desc        = Desc;                       ///< The descriptor of the layer
    using this_type   = dyn_lcn_layer_impl<Desc>;   ///< This layer's type
    using base_type   = transform_layer<this_type>; ///< The base type
    using layer_t     = this_type;                  ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t; ///< The dynamic version of this layer

    size_t K;
    size_t Mid;
    double sigma = 2.0;

    /*!
     * \brief Initialize the dynamic layer
     */
    void init_layer(size_t K){
        cpp_assert(K > 1, "The LCN kernel size must be greater than 1");
        cpp_assert(K % 2 == 1, "The LCN kernel size must be odd");

        this->K = K;
        this->Mid = K / 2;
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    std::string to_short_string([[maybe_unused]] std::string pre = "") const {
        std::string desc("LCN(dyn): ");
        desc += std::to_string(K) + 'x' + std::to_string(K);
        return desc;
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    std::string to_full_string([[maybe_unused]] std::string pre = "") const {
        std::string desc("LCN(dyn): ");
        desc += std::to_string(K) + 'x' + std::to_string(K);
        return desc;
    }

    template <typename W>
    etl::dyn_matrix<W, 2> filter(double sigma) const {
        etl::dyn_matrix<W, 2> w(K, K);

        lcn_filter(w, K, Mid, sigma);

        return w;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void forward_batch(Output&& output, Input&& input) const {
        inherit_dim(output, input);

        using weight_t = etl::value_t<Input>;

        auto w = filter<weight_t>(sigma);

        for (size_t b = 0; b < etl::dim<0>(input); ++b) {
            lcn_compute(output(b), input(b), w, K, Mid);
        }
    }
};

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<dyn_lcn_layer_impl<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = true;  ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = true; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dyn_lcn_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_lcn_layer_impl<Desc>, L> {
    using layer_t          = dyn_lcn_layer_impl<Desc>;                            ///< The current layer type
    using previous_layer   = typename DBN::template layer_type<L - 1>;          ///< The previous layer type
    using previous_context = sgd_context<DBN, previous_layer, L - 1>;           ///< The previous layer's context
    using inputs_t         = decltype(std::declval<previous_context>().output); ///< The type of inputs

    inputs_t input;  ///< A batch of input
    inputs_t output; ///< A batch of output
    inputs_t errors; ///< A batch of errors

    sgd_context(const layer_t& /*layer*/){}
};

} //end of dll namespace
