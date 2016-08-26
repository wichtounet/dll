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

    /*!
     * \brief Display the layer to the console
     */
    void display() const {
        std::cout << to_short_string() << std::endl;
    }

    using input_one_t  = typename base::input_one_t;  ///< The type of one input
    using output_one_t = typename base::output_one_t; ///< The type of one output
    using input_t      = typename base::input_t;      ///< The type of many input
    using output_t     = typename base::output_t;     ///< The type of many output

    template <std::size_t B>
    using input_batch_t = typename base::template input_batch_t<B>;

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
};

} //end of dll namespace
