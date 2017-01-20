//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "pooling_layer.hpp"

namespace dll {

/*!
 * \brief Standard dyn max pooling layer
 */
template <typename Desc>
struct dyn_mp_layer_3d final : dyn_pooling_layer_3d<dyn_mp_layer_3d<Desc>, Desc> {
    using desc      = Desc;                                  ///< The layer descriptor
    using weight    = typename desc::weight;                 ///< The layer weight type
    using this_type = dyn_mp_layer_3d<Desc>;                 ///< This layer's type
    using base      = dyn_pooling_layer_3d<this_type, desc>; ///< The layer base type

    dyn_mp_layer_3d() = default;

    /*!
     * \brief Get a string representation of the layer
     */
    std::string to_short_string() const {
        char buffer[1024];
        snprintf(buffer, 1024, "MP(3D): %lux%lux%lu -> (%lux%lux%lu) -> %lux%lux%lu",
                 base::i1, base::i2, base::i3, base::c1, base::c2, base::c3, base::o1, base::o2, base::o3);
        return {buffer};
    }

    using input_one_t  = typename base::input_one_t;  ///< The type of one input
    using output_one_t = typename base::output_one_t; ///< The type of one output
    using input_t      = typename base::input_t;      ///< The type of many input
    using output_t     = typename base::output_t;     ///< The type of many output

    void activate_hidden(output_one_t& h, const input_one_t& v) const {
        h = etl::max_pool_3d(v, base::c1, base::c2, base::c3);
    }

    template <typename Input, typename Output>
    void batch_activate_hidden(Output& output, const Input& input) const {
        for (std::size_t b = 0; b < etl::dim<0>(input); ++b) {
            output(b) = etl::max_pool_3d(input(b), base::c1, base::c2, base::c3);
        }
    }

    template <typename DBN>
    void init_sgd_context() {
        this->sgd_context_ptr = std::make_shared<sgd_context<DBN, this_type>>(base::i1, base::i2, base::i3, base::c1, base::c2, base::c3);
    }

    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
    }

    template<typename C>
    void adapt_errors(C& context) const {
        cpp_unused(context);
    }

    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        size_t c1 = base::c1;
        size_t c2 = base::c2;
        size_t c3 = base::c3;

        const auto batch_size = etl::dim<0>(context.input);

        // TODO The derivative should handle batch
        for (std::size_t i = 0; i < batch_size; ++i) {
            output(i) = etl::max_pool_derivative_3d(context.input(i), context.output(i), c1, c2, c3) >> etl::upsample_3d(context.errors(i), c1, c2, c3);
        }
    }

    template<typename C>
    void compute_gradients(C& context) const {
        cpp_unused(context);
    }
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<dyn_mp_layer_3d<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = true;  ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = false; ///< Indicates if the layer is a patches layer
    static constexpr bool is_augment    = false; ///< Indicates if the layer is an augment layer
    static constexpr bool is_activation = false; ///< Indicates if the layer is an activation-only layer
    static constexpr bool is_dynamic    = true; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

} //end of dll namespace
