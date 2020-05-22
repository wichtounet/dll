//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
struct dyn_mp_2d_layer_impl final : dyn_pooling_2d_layer<dyn_mp_2d_layer_impl<Desc>, Desc> {
    using desc        = Desc;                                  ///< The layer descriptor
    using weight      = typename desc::weight;                 ///< The layer weight type
    using this_type   = dyn_mp_2d_layer_impl<Desc>;            ///< This layer's type
    using base        = dyn_pooling_2d_layer<this_type, desc>; ///< The layer base type
    using layer_t     = this_type;                             ///< The type of this layer
    using dyn_layer_t = typename desc::dyn_layer_t;            ///< The dynamic type of this layer

    using input_one_t  = typename base::input_one_t;  ///< The type of one input
    using output_one_t = typename base::output_one_t; ///< The type of one output
    using input_t      = typename base::input_t;      ///< The type of many input
    using output_t     = typename base::output_t;     ///< The type of many output

    dyn_mp_2d_layer_impl() = default;

    /*!
     * \brief Get a string representation of the layer
     */
    std::string to_short_string([[maybe_unused]] std::string pre = "") const {
        return "MP(2D)";
    }

    /*!
     * \brief Get a string representation of the layer
     */
    std::string to_full_string([[maybe_unused]] std::string pre = "") const {
        char buffer[1024];
        snprintf(buffer, 1024, "MP(2d): %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu",
                 base::i1, base::i2, base::i3, base::c1, base::c2, base::o1, base::o2, base::o3);
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
        dll::auto_timer timer("mp:forward_batch");

        output = etl::ml::max_pool_forward(input, base::c1, base::c2);
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
        dll::auto_timer timer("mp:backward_batch");

        size_t c1 = base::c1;
        size_t c2 = base::c2;

        output = etl::ml::max_pool_backward(context.input, context.output, context.errors, c1, c2);
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
struct layer_base_traits<dyn_mp_2d_layer_impl<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = true;  ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = true; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dyn_mp_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_mp_2d_layer_impl<Desc>, L> {
    using layer_t = dyn_mp_2d_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::dyn_matrix<weight, 4> input;
    etl::dyn_matrix<weight, 4> output;
    etl::dyn_matrix<weight, 4> errors;

    sgd_context(const layer_t& layer)
            : input(batch_size, layer.i1, layer.i2, layer.i3),
              output(batch_size, layer.i1, layer.i2 / layer.c1, layer.i3 / layer.c2),
              errors(batch_size, layer.i1, layer.i2 / layer.c1, layer.i3 / layer.c2) {}
};

/*!
 * \brief Standard dyn max pooling layer
 */
template <typename Desc>
struct dyn_mp_3d_layer_impl final : dyn_pooling_3d_layer<dyn_mp_3d_layer_impl<Desc>, Desc> {
    using desc        = Desc;                                  ///< The layer descriptor
    using weight      = typename desc::weight;                 ///< The layer weight type
    using this_type   = dyn_mp_3d_layer_impl<Desc>;            ///< This layer's type
    using base        = dyn_pooling_3d_layer<this_type, desc>; ///< The layer base type
    using layer_t     = this_type;                             ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;            ///< The dynamic version of this layer

    using input_one_t  = typename base::input_one_t;  ///< The type of one input
    using output_one_t = typename base::output_one_t; ///< The type of one output
    using input_t      = typename base::input_t;      ///< The type of many input
    using output_t     = typename base::output_t;     ///< The type of many output

    dyn_mp_3d_layer_impl() = default;

    /*!
     * \brief Get a string representation of the layer
     */
    std::string to_short_string([[maybe_unused]] std::string pre = "") const {
        return "MP(3D)";
    }

    /*!
     * \brief Get a string representation of the layer
     */
    std::string to_full_string([[maybe_unused]] std::string pre = "") const {
        char buffer[1024];
        snprintf(buffer, 1024, "MP(3D): %lux%lux%lu -> (%lux%lux%lu) -> %lux%lux%lu",
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
     * \brief Return the size, in bytes, used by this layer
     * \return the size, in bytes, used by this layer
     */
    size_t memory_size() const noexcept {
        return 0;
    }

    /*!
     * \brief Return the size, in bytes, used by the context of this layer
     * \return the size, in bytes, used by the context of this layer
     */
    size_t context_memory_size(size_t batch_size) const noexcept {
        return batch_size * base::i1 * base::i2 * base::i3                                           // Input
               + batch_size * (base::i1 / base::c1) * (base::i2 / base::c2) * (base::i3 / base::c3)  // Output
               + batch_size * (base::i1 / base::c1) * (base::i2 / base::c2) * (base::i3 / base::c3); // Errors
    }

    /*!
     * \brief Forward activation of the layer for one batch of sample
     * \param output The output matrix
     * \param input The input matrix
     */
    template <typename Input, typename Output>
    void forward_batch(Output& output, const Input& input) const {
        output = etl::ml::max_pool_3d_forward(input, base::c1, base::c2, base::c3);
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
        size_t c1 = base::c1;
        size_t c2 = base::c2;
        size_t c3 = base::c3;

        output = etl::ml::max_pool_3d_backward(context.input, context.output, context.errors, c1, c2, c3);
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
struct layer_base_traits<dyn_mp_3d_layer_impl<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = true;  ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = true; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dyn_mp_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_mp_3d_layer_impl<Desc>, L> {
    using layer_t = dyn_mp_3d_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::dyn_matrix<weight, 4> input;
    etl::dyn_matrix<weight, 4> output;
    etl::dyn_matrix<weight, 4> errors;

    sgd_context(const layer_t& layer)
            : input(batch_size, layer.i1, layer.i2, layer.i3),
              output(batch_size, layer.i1 / layer.c1, layer.i2 / layer.c2, layer.i3 / layer.c3),
              errors(batch_size, layer.i1 / layer.c1, layer.i2 / layer.c2, layer.i3 / layer.c3) {}
};

} //end of dll namespace
