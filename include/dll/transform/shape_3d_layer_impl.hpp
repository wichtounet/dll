//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/transform/transform_layer.hpp"

namespace dll {

/*!
 * \brief Simple shape information layer
 */
template <typename Desc>
struct shape_3d_layer_impl : transform_layer<shape_3d_layer_impl<Desc>> {
    using desc      = Desc;                       ///< The descriptor type
    using weight    = typename desc::weight;      ///< The data type
    using this_type = shape_3d_layer_impl<desc>;       ///< The type of this layer
    using base_type = transform_layer<this_type>; ///< The base type
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    static constexpr size_t D = 3;       ///< The number of dimensions
    static constexpr size_t C = desc::C; ///< The number of channels
    static constexpr size_t W = desc::W; ///< The height of the input
    static constexpr size_t H = desc::H; ///< The width of the input

    using input_one_t  = etl::fast_dyn_matrix<weight, C, W, H>; ///< The preferred type of input
    using output_one_t = etl::fast_dyn_matrix<weight, C, W, H>; ///< The type of output

    shape_3d_layer_impl() = default;

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        return "Shape3d";
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        return "Shape3d";
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {C, W, H};
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    static constexpr size_t input_size() {
        return C * W * H;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    static constexpr size_t output_size() {
        return C * W * H;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void forward_batch(Output& output, const Input& input) {
        output = input;
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
    template <typename H, typename C>
    void backward_batch([[maybe_unused]] H&& output, [[maybe_unused]] C& context) const {}

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template <typename C>
    void compute_gradients([[maybe_unused]] C& context) const {}
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const size_t shape_3d_layer_impl<Desc>::C;

template <typename Desc>
const size_t shape_3d_layer_impl<Desc>::H;

template <typename Desc>
const size_t shape_3d_layer_impl<Desc>::W;

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<shape_3d_layer_impl<Desc>> {
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
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for lcn_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, shape_3d_layer_impl<Desc>, L> {
    using layer_t = shape_3d_layer_impl<Desc>;
    using weight  = typename DBN::weight; ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, layer_t::C, layer_t::H, layer_t::W> input;
    etl::fast_matrix<weight, batch_size, layer_t::C, layer_t::H, layer_t::W> output;
    etl::fast_matrix<weight, batch_size, layer_t::C, layer_t::H, layer_t::W> errors;

    sgd_context(const shape_3d_layer_impl<Desc>& /* layer */){}
};

} //end of dll namespace
