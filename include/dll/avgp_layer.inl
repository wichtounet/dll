//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "pooling_layer.hpp"

namespace dll {

/*!
 * \brief Standard average pooling layer
 */
template <typename Desc>
struct avgp_layer_3d final : pooling_layer_3d<avgp_layer_3d<Desc>, Desc> {
    using desc   = Desc;
    using weight = typename desc::weight;
    using base   = pooling_layer_3d<avgp_layer_3d<Desc>, desc>;

    using input_one_t  = typename base::input_one_t;
    using output_one_t = typename base::output_one_t;
    using input_t      = typename base::input_t;
    using output_t     = typename base::output_t;

    template <std::size_t B>
    using input_batch_t = typename base::template input_batch_t<B>;

    template <std::size_t B>
    using output_batch_t = typename base::template output_batch_t<B>;

    avgp_layer_3d() = default;

    static std::string to_short_string() {
        char buffer[1024];
        snprintf(buffer, 1024, "MP(3D): %lux%lux%lu -> (%lux%lux%lu) -> %lux%lux%lu",
                 base::I1, base::I2, base::I3, base::C1, base::C2, base::C3, base::O1, base::O2, base::O3);
        return {buffer};
    }

    static void display() {
        std::cout << to_short_string() << std::endl;
    }

    static void activate_hidden(output_one_t& h, const input_one_t& v) {
        h = etl::avg_pool_3d<base::C1, base::C2, base::C3>(v);
    }

    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        for (std::size_t b = 0; b < etl::dim<0>(input); ++b) {
            output(b) = etl::avg_pool_3d<base::C1, base::C2, base::C3>(input(b));
        }
    }

    static void activate_many(output_t& h_a, const input_t& input) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            activate_one(input[i], h_a[i]);
        }
    }
};

} //end of dll namespace
