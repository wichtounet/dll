//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "unpooling_layer.hpp"

namespace dll {

/*!
 * \brief Standard dyn upsample layer
 */
template <typename Desc>
struct dyn_upsample_3d_layer_impl final : dyn_unpooling_3d_layer<dyn_upsample_3d_layer_impl<Desc>, Desc> {
    using desc      = Desc;                                    ///< The layer descriptor
    using weight    = typename desc::weight;                   ///< The layer weight type
    using this_type = dyn_upsample_3d_layer_impl<Desc>;             ///< This layer's type
    using base      = dyn_unpooling_3d_layer<this_type, desc>; ///< The layer base type
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    using input_one_t  = typename base::input_one_t;  ///< The type of one input
    using output_one_t = typename base::output_one_t; ///< The type of one output
    using input_t      = typename base::input_t;      ///< The type of many input
    using output_t     = typename base::output_t;     ///< The type of many output

    dyn_upsample_3d_layer_impl() = default;

    /*!
     * \brief Get a string representation of the layer
     */
    std::string to_short_string([[maybe_unused]] std::string pre = "") const {
        return "upsample(3D)";
    }

    /*!
     * \brief Get a string representation of the layer
     */
    std::string to_full_string([[maybe_unused]] std::string pre = "") const {
        char buffer[512];
        snprintf(buffer, 512, "upsample(3D): %lux%lux%lu -> (%lux%lux%lu) -> %lux%lux%lu",
                 base::i1, base::i2, base::i3, base::c1, base::c2, base::c3, base::o1, base::o2, base::o3);
        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {base::o1, base::o2, base::o3};
    }

    /*!
     * \brief Forward activation of the layer for one batch of sample
     * \param output The output matrix
     * \param input The input matrix
     */
    template <typename Input, typename Output>
    void forward_batch(Output& output, const Input& input) const {
        output = etl::upsample_3d(input, base::c1, base::c2, base::c3);
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

    /*!
     * \brief Adapt the errors, called before backpropagation of the errors.
     *
     * This must be used by layers that have both an activation fnction and a non-linearity.
     *
     * \param context the training context
     */
    template <typename C>
    void adapt_errors([[maybe_unused]] C& context) const {}

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        const size_t c1 = base::c1;
        const size_t c2 = base::c2;
        const size_t c3 = base::c3;

        if constexpr (etl::decay_traits<H>::dimensions() == 4) {
            output = etl::max_pool_3d(context.errors, c1, c2, c3);
        } else {
            const size_t B = etl::dim<0>(output);

            etl::reshape(output, B, base::i1, base::i2, base::i3) = etl::max_pool_3d(context.errors, c1, c2, c3);
        }
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template <typename C>
    void compute_gradients([[maybe_unused]] C& context) const {}
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<dyn_upsample_3d_layer_impl<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = true;  ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = true;  ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dyn_upsample_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_upsample_3d_layer_impl<Desc>, L> {
    using layer_t = dyn_upsample_3d_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::dyn_matrix<weight, 4> input;
    etl::dyn_matrix<weight, 4> output;
    etl::dyn_matrix<weight, 4> errors;

    sgd_context(const layer_t& layer)
            : input(batch_size, layer.i1, layer.i2, layer.i3),
              output(batch_size, layer.i1 * layer.c1, layer.i2 * layer.c2, layer.i3 * layer.c3),
              errors(batch_size, layer.i1 * layer.c1, layer.i2 * layer.c2, layer.i3 * layer.c3) {}
};

} //end of dll namespace
