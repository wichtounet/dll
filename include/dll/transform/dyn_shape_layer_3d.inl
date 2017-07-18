//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "transform_layer.hpp"
#include "lcn.hpp"

namespace dll {

template <typename Desc>
struct dyn_shape_layer_3d : transform_layer<dyn_shape_layer_3d<Desc>> {
    using desc      = Desc;                       ///< The descriptor type
    using weight    = typename desc::weight;      ///< The data type
    using this_type = dyn_shape_layer_3d<desc>;   ///< The type of this layer
    using base_type = transform_layer<this_type>; ///< The base type

    static constexpr size_t D = 3; ///< The number of dimensions

    using input_one_t  = etl::dyn_matrix<weight, 3>; ///< The preferred type of input
    using output_one_t = etl::dyn_matrix<weight, 3>; ///< The type of output

    size_t C; ///< The number of input channels
    size_t W; ///< The width of the input
    size_t H; ///< The height of the input

    /*!
     * \brief Initialize the dynamic layer
     */
    void init_layer(size_t C, size_t W, size_t H){
        cpp_assert(C > 1, "The shape must be bigger than 0");
        cpp_assert(W > 1, "The shape must be bigger than 0");
        cpp_assert(H > 1, "The shape must be bigger than 0");

        this->C = C;
        this->W = W;
        this->H = H;
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "Shape3d(dyn)";
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    size_t input_size() const  {
        return C * W * H;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    size_t output_size() const {
        return C * W * H;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void batch_activate_hidden(Output& output, const Input& input) const {
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

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<dyn_shape_layer_3d<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = true;  ///< Indicates if the layer is a transform layer
    static constexpr bool is_dynamic    = true; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dyn_lcn_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_shape_layer_3d<Desc>, L> {
    using layer_t = dyn_shape_layer_3d<Desc>;
    using weight  = typename DBN::weight; ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    using inputs_t = etl::dyn_matrix<weight, 4>;

    inputs_t input;  ///< A batch of input
    inputs_t output; ///< A batch of output
    inputs_t errors; ///< A batch of errors

    sgd_context(layer_t& layer) : input(batch_size, layer.C, layer.W, layer.H), output(batch_size, layer.C, layer.W, layer.H), errors(batch_size, layer.C, layer.W, layer.H){}
};

} //end of dll namespace
