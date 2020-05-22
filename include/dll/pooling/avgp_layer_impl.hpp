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
 * \brief Standard average pooling layer
 */
template <typename Desc>
struct avgp_2d_layer_impl final : pooling_2d_layer<avgp_2d_layer_impl<Desc>, Desc> {
    using desc        = Desc;                                             ///< The layer descriptor
    using weight      = typename desc::weight;                            ///< The layer weight type
    using base        = pooling_2d_layer<avgp_2d_layer_impl<Desc>, desc>; ///< The layer base type
    using this_type   = avgp_2d_layer_impl<Desc>;                         ///< The type of this layer
    using layer_t     = this_type;                                        ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;                       ///< The dynamic version of this layer

    using input_one_t  = typename base::input_one_t;  ///< The type of one input
    using output_one_t = typename base::output_one_t; ///< The type of one output
    using input_t      = typename base::input_t;      ///< The type of many input
    using output_t     = typename base::output_t;     ///< The type of many output

    avgp_2d_layer_impl() = default;

    /*!
     * \brief Get a string representation of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        return "AVGP(2D)";
    }

    /*!
     * \brief Get a string representation of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        char buffer[1024];
        snprintf(buffer, 1024, "AVGP(2d): %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu",
                 base::I1, base::I2, base::I3, base::C1, base::C2, base::O1, base::O2, base::O3);
        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {base::O1, base::O2, base::O3};
    }

    /*!
     * \brief Forward activation of the layer for one batch of sample
     * \param output The output matrix
     * \param input The input matrix
     */
    template <typename Input, typename Output>
    static void forward_batch(Output& output, const Input& input) {
        output = etl::ml::avg_pool_forward<base::C1, base::C2>(input);
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DLayer>
    static void dyn_init(DLayer& dyn){
        dyn.init_layer(base::I1, base::I2, base::I3, base::C1, base::C2);
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
        static constexpr size_t C1 = base::C1; ///< The pooling first dimension
        static constexpr size_t C2 = base::C2; ///< The pooling second dimension
        static constexpr size_t C3 = base::C3; ///< The pooling second dimension

        output = etl::ml::avg_pool_backward<C1, C2, C3>(context.input, context.output, context.errors);
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
struct layer_base_traits<avgp_2d_layer_impl<Desc>> {
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
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for mp_2d_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, avgp_2d_layer_impl<Desc>, L> {
    using layer_t = avgp_2d_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr size_t I1 = layer_t::I1; ///< The input first dimension
    static constexpr size_t I2 = layer_t::I2; ///< The input second dimension
    static constexpr size_t I3 = layer_t::I3; ///< The input third dimension

    static constexpr size_t O1 = layer_t::O1; ///< The padding first dimension
    static constexpr size_t O2 = layer_t::O2; ///< The padding second dimension
    static constexpr size_t O3 = layer_t::O3; ///< The padding third dimension

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, I1, I2, I3> input;
    etl::fast_matrix<weight, batch_size, O1, O2, O3> output;
    etl::fast_matrix<weight, batch_size, O1, O2, O3> errors;

    sgd_context(const avgp_2d_layer_impl<Desc>& /*layer*/){}
};

/*!
 * \brief Standard average pooling layer
 */
template <typename Desc>
struct avgp_3d_layer_impl final : pooling_3d_layer<avgp_3d_layer_impl<Desc>, Desc> {
    using desc        = Desc;                                             ///< The layer descriptor
    using weight      = typename desc::weight;                            ///< The layer weight type
    using base        = pooling_3d_layer<avgp_3d_layer_impl<Desc>, desc>; ///< The layer base type
    using this_type   = avgp_3d_layer_impl<Desc>;                         ///< The type of this layer
    using layer_t     = this_type;                                        ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;                       ///< The dynamic version of this layer

    using input_one_t  = typename base::input_one_t;  ///< The type of one input
    using output_one_t = typename base::output_one_t; ///< The type of one output
    using input_t      = typename base::input_t;      ///< The type of many input
    using output_t     = typename base::output_t;     ///< The type of many output

    avgp_3d_layer_impl() = default;

    /*!
     * \brief Get a string representation of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        return "AVGP(3D)";
    }

    /*!
     * \brief Get a string representation of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        char buffer[1024];
        snprintf(buffer, 1024, "AVGP(3D): %lux%lux%lu -> (%lux%lux%lu) -> %lux%lux%lu",
                 base::I1, base::I2, base::I3, base::C1, base::C2, base::C3, base::O1, base::O2, base::O3);
        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {base::O1, base::O2, base::O3};
    }

    /*!
     * \brief Forward activation of the layer for one batch of sample
     * \param output The output matrix
     * \param input The input matrix
     */
    template <typename Input, typename Output>
    static void forward_batch(Output& output, const Input& input) {
        output = etl::ml::avg_pool_3d_forward<base::C1, base::C2, base::C3>(input);
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DLayer>
    static void dyn_init(DLayer& dyn){
        dyn.init_layer(base::I1, base::I2, base::I3, base::C1, base::C2, base::C3);
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
        static constexpr size_t C1 = base::C1; ///< The pooling first dimension
        static constexpr size_t C2 = base::C2; ///< The pooling second dimension
        static constexpr size_t C3 = base::C3; ///< The pooling third dimension

        output = etl::ml::avg_pool_3d_backward<C1, C2, C3>(context.input, context.output, context.errors);
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
struct layer_base_traits<avgp_3d_layer_impl<Desc>> {
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
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for mp_3d_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, avgp_3d_layer_impl<Desc>, L> {
    using layer_t = avgp_3d_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr size_t I1 = layer_t::I1; ///< The input first dimension
    static constexpr size_t I2 = layer_t::I2; ///< The input second dimension
    static constexpr size_t I3 = layer_t::I3; ///< The input third dimension

    static constexpr size_t O1 = layer_t::O1; ///< The padding first dimension
    static constexpr size_t O2 = layer_t::O2; ///< The padding second dimension
    static constexpr size_t O3 = layer_t::O3; ///< The padding third dimension

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, I1, I2, I3> input;
    etl::fast_matrix<weight, batch_size, O1, O2, O3> output;
    etl::fast_matrix<weight, batch_size, O1, O2, O3> errors;

    sgd_context(const avgp_3d_layer_impl<Desc>& /*layer*/){}
};

} //end of dll namespace
