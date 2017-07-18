//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
struct shape_layer_3d : transform_layer<shape_layer_3d<Desc>> {
    using desc      = Desc;                       ///< The descriptor type
    using weight    = typename desc::weight;      ///< The data type
    using this_type = shape_layer_3d<desc>;       ///< The type of this layer
    using base_type = transform_layer<this_type>; ///< The base type

    static constexpr size_t D = 3;       ///< The number of dimensions
    static constexpr size_t C = desc::C; ///< The number of channels
    static constexpr size_t W = desc::W; ///< The height of the input
    static constexpr size_t H = desc::H; ///< The width of the input

    using input_one_t  = etl::fast_dyn_matrix<weight, C, W, H>; ///< The preferred type of input
    using output_one_t = etl::fast_dyn_matrix<weight, C, W, H>; ///< The type of output

    shape_layer_3d() = default;

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "Shape3d";
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
    static void batch_activate_hidden(Output& output, const Input& input) {
        output = input;
    }

    /*!
     * \brief Adapt the errors, called before backpropagation of the errors.
     *
     * This must be used by layers that have both an activation fnction and a non-linearity.
     *
     * \param context the training context
     */
    template<typename C>
    void adapt_errors(C& context) const {
        cpp_unused(context);
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        cpp_unused(output);
        cpp_unused(context);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        cpp_unused(context);
    }
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const size_t shape_layer_3d<Desc>::C;

template <typename Desc>
const size_t shape_layer_3d<Desc>::H;

template <typename Desc>
const size_t shape_layer_3d<Desc>::W;

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<shape_layer_3d<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = true;  ///< Indicates if the layer is a transform layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for lcn_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, shape_layer_3d<Desc>, L> {
    using layer_t = shape_layer_3d<Desc>;
    using weight  = typename DBN::weight; ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, layer_t::C, layer_t::H, layer_t::W> input;
    etl::fast_matrix<weight, batch_size, layer_t::C, layer_t::H, layer_t::W> output;
    etl::fast_matrix<weight, batch_size, layer_t::C, layer_t::H, layer_t::W> errors;

    sgd_context(const shape_layer_3d<Desc>& /* layer */){}
};

} //end of dll namespace
