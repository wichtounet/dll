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
struct shape_layer_1d : transform_layer<shape_layer_1d<Desc>> {
    using desc      = Desc;                       ///< The descriptor type
    using weight    = typename desc::weight;      ///< The data type
    using this_type = shape_layer_1d<desc>;       ///< The type of this layer
    using base_type = transform_layer<this_type>; ///< The base type

    static constexpr size_t Size = desc::S; ///< The input size
    static constexpr size_t D    = 1;       ///< The number of dimensions

    using input_one_t  = etl::fast_dyn_matrix<weight, Size>; ///< The preferred type of input
    using output_one_t = etl::fast_dyn_matrix<weight, Size>; ///< The type of output

    shape_layer_1d() = default;

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "Shape";
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    static constexpr size_t input_size() {
        return Size;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    static constexpr size_t output_size() {
        return Size;
    }

    using base_type::activate_hidden;

    /*!
     * \brief Apply the layer to the input
     * \param output The output
     * \param input The input to apply the layer to
     */
    template <typename Input, typename Output>
    static void activate_hidden(Output& output, const Input& input) {
        output = input;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \return A batch of output corresponding to the activated input
     */
    template <typename V, cpp_enable_if((etl::decay_traits<V>::is_fast))>
    auto batch_activate_hidden(const V& v) const {
        static constexpr auto Batch = etl::decay_traits<V>::template dim<0>();

        etl::fast_dyn_matrix<etl::value_t<V>, Batch, Size> output;
        batch_activate_hidden(output, v);
        return output;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \return A batch of output corresponding to the activated input
     */
    template <typename V, cpp_disable_if((etl::decay_traits<V>::is_fast))>
    auto batch_activate_hidden(const V& v) const {
        const auto Batch = etl::dim<0>(v);

        etl::dyn_matrix<etl::value_t<V>, 2> output(Batch, Size);
        batch_activate_hidden(output, v);
        return output;
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
const size_t shape_layer_1d<Desc>::Size;

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<shape_layer_1d<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = true;  ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = false; ///< Indicates if the layer is a patches layer
    static constexpr bool is_augment    = false; ///< Indicates if the layer is an augment layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for shape_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, shape_layer_1d<Desc>, L> {
    using layer_t = shape_layer_1d<Desc>;
    using weight  = typename DBN::weight;

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, layer_t::Size> input;
    etl::fast_matrix<weight, batch_size, layer_t::Size> output;
    etl::fast_matrix<weight, batch_size, layer_t::Size> errors;

    sgd_context(const shape_layer_1d<Desc>& /* layer */){}
};

} //end of dll namespace
