//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/layer.hpp"

namespace dll {

/*!
 * \brief Layer to cut images into patches.
 */
template <typename Desc>
struct dyn_patches_layer_padh : layer<dyn_patches_layer_padh<Desc>> {
    using desc = Desc;
    using base_type = layer<dyn_patches_layer_padh<Desc>>; ///< The base type

    using weight = typename desc::weight;

    using input_one_t  = etl::dyn_matrix<weight, 3>;
    using input_t      = std::vector<input_one_t>;

    using output_one_t  = std::vector<etl::dyn_matrix<weight, 3>>;
    using output_t      = std::vector<output_one_t>;

    size_t width;
    size_t height;
    size_t v_stride;
    size_t h_stride;
    size_t filler;
    size_t h_context;

    dyn_patches_layer_padh() = default;

    void init_layer(size_t width, size_t height, size_t v_stride, size_t h_stride, size_t filler) {
        this->width     = width;
        this->height    = height;
        this->v_stride  = v_stride;
        this->h_stride  = h_stride;
        this->filler    = filler;
        this->h_context = width / 2;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_short_string() const {
        char buffer[1024];
        snprintf(buffer, 1024, "Patches(padh,dyn) -> (%lu:%lux%lu:%lu)", height, v_stride, width, h_stride);
        return {buffer};
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    size_t output_size() const noexcept {
        return width * height;
    }

    using base_type::activate_hidden;

    template<typename Input>
    void activate_hidden(output_one_t& h_a, const Input& input) const {
        cpp_assert(etl::dim<0>(input) == 1, "Only one channel is supported for now");

        h_a.clear();

        for (size_t y = 0; y + height <= etl::dim<1>(input); y += v_stride) {
            for (size_t x = 0; x < etl::dim<2>(input); x += h_stride) {
                h_a.emplace_back();

                auto& patch = h_a.back();

                patch.inherit_if_null(etl::dyn_matrix<weight,3>(1UL, height, width));

                for (size_t yy = 0; yy < height; ++yy) {
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
        for (size_t i = 0; i < input.size(); ++i) {
            activate_one(input[i], h_a[i]);
        }
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    static output_t prepare_output(size_t samples) {
        return output_t(samples);
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    static output_one_t prepare_one_output() {
        return output_one_t();
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
    }
};

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<dyn_patches_layer_padh<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false;  ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = true; ///< Indicates if the layer is a patches layer
    static constexpr bool is_dynamic    = true; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

} //end of dll namespace
