//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "unpooling_layer.hpp"

namespace dll {

/*!
 * \brief Standard max pooling layer
 */
template <typename Desc>
struct upsample_layer_3d final : unpooling_layer_3d<upsample_layer_3d<Desc>, Desc> {
    using desc   = Desc;                                      ///< The layer descriptor
    using weight = typename desc::weight;                     ///< The layer weight type
    using base   = unpooling_layer_3d<upsample_layer_3d<Desc>, desc>; ///< The layer base type

    upsample_layer_3d() = default;

    /*!
     * \brief Get a string representation of the layer
     */
    static std::string to_short_string() {
        char buffer[1024];
        snprintf(buffer, 1024, "upsample(3D): %lux%lux%lu -> (%lux%lux%lu) -> %lux%lux%lu",
                 base::I1, base::I2, base::I3, base::C1, base::C2, base::C3, base::O1, base::O2, base::O3);
        return {buffer};
    }

    using input_one_t  = typename base::input_one_t;  ///< The type of one input
    using output_one_t = typename base::output_one_t; ///< The type of one output
    using input_t      = typename base::input_t;      ///< The type of many input
    using output_t     = typename base::output_t;     ///< The type of many output

    static void activate_hidden(output_one_t& h, const input_one_t& v) {
        h = etl::upsample_3d<base::C1, base::C2, base::C3>(v);
    }

    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        for (std::size_t b = 0; b < etl::dim<0>(input); ++b) {
            output(b) = etl::upsample_3d<base::C1, base::C2, base::C3>(input(b));
        }
    }

    template<typename DLayer>
    static void dyn_init(DLayer& dyn){
        cpp_unused(dyn);
        //TODO dyn.init_layer(base::I1, base::I2, base::I3, base::C1, base::C2, base::C3);
    }

    template<typename C>
    void adapt_errors(C& context) const {
        cpp_unused(context);
    }

    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        static constexpr size_t C1 = base::C1;
        static constexpr size_t C2 = base::C2;
        static constexpr size_t C3 = base::C3;

        constexpr const auto batch_size = etl::decay_traits<H>::template dim<0>();

        // TODO The derivative should handle batch
        for (std::size_t i = 0; i < batch_size; ++i) {
            output(i) = etl::max_pool_3d<C1, C2, C3>(context.errors(i));
        }
    }

    template<typename C>
    void compute_gradients(C& context) const {
        cpp_unused(context);
    }
};

} //end of dll namespace
