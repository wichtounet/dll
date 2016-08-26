//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "neural_base.hpp"

namespace dll {

/*!
 * \brief Layer to cut images into patches.
 */
template <typename Desc>
struct dyn_patches_layer_padh : neural_base<dyn_patches_layer_padh<Desc>> {
    using desc = Desc;

    using weight = typename desc::weight;

    using input_one_t  = etl::dyn_matrix<weight, 3>;
    using input_t      = std::vector<input_one_t>;

    template <std::size_t B>
    using input_batch_t = etl::fast_dyn_matrix<weight, 1, 1, 1, 1>; //This is fake, should never be used

    using output_one_t  = std::vector<etl::dyn_matrix<weight, 3>>;
    using output_t      = std::vector<output_one_t>;

    std::size_t width;
    std::size_t height;
    std::size_t v_stride;
    std::size_t h_stride;
    std::size_t filler;
    std::size_t h_context;

    dyn_patches_layer_padh() = default;

    void init_layer(std::size_t width, std::size_t height, std::size_t v_stride, std::size_t h_stride, std::size_t filler) {
        this->width     = width;
        this->height    = height;
        this->v_stride  = v_stride;
        this->h_stride  = h_stride;
        this->filler    = filler;
        this->h_context = width / 2;
    }

    std::string to_short_string() const {
        char buffer[1024];
        snprintf(buffer, 1024, "Patches(padh,dyn) -> (%lu:%lux%lu:%lu)", height, v_stride, width, h_stride);
        return {buffer};
    }

    void display() const {
        std::cout << to_short_string() << std::endl;
    }

    std::size_t output_size() const noexcept {
        return width * height;
    }

    void activate_hidden(output_one_t& h_a, const input_one_t& input) const {
        cpp_assert(etl::dim<0>(input) == 1, "Only one channel is supported for now");

        h_a.clear();

        for (std::size_t y = 0; y + height <= etl::dim<1>(input); y += v_stride) {
            for (std::size_t x = 0; x < etl::dim<2>(input); x += h_stride) {
                h_a.emplace_back();

                auto& patch = h_a.back();

                patch.inherit_if_null(etl::dyn_matrix<weight,3>(1UL, height, width));

                for (std::size_t yy = 0; yy < height; ++yy) {
                    for (int xx = x - h_context; xx < int(x + h_context); ++xx) {
                        double value(filler);

                        if (xx >= 0 && xx < int(etl::dim<2>(input))) {
                            value = input(0, y + yy, xx);
                        }

                        patch(0, yy, xx - x + h_context) = value;
                    }
                }
            }
        }
    }

    void activate_many(output_t& h_a, const input_t& input) const {
        for (std::size_t i = 0; i < input.size(); ++i) {
            activate_one(input[i], h_a[i]);
        }
    }

    template <typename Input>
    static output_t prepare_output(std::size_t samples) {
        return output_t(samples);
    }

    template <typename Input>
    static output_one_t prepare_one_output() {
        return output_one_t();
    }

    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
    }
};

} //end of dll namespace
