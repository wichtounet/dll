//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
struct dyn_shape_1d_layer_impl : transform_layer<dyn_shape_1d_layer_impl<Desc>> {
    using desc      = Desc;                       ///< The descriptor type
    using weight    = typename desc::weight;      ///< The data type
    using this_type = dyn_shape_1d_layer_impl<desc>;   ///< The type of this layer
    using base_type = transform_layer<this_type>; ///< The base type
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    static constexpr size_t D = 1; ///< The number of dimensions

    using input_one_t  = etl::dyn_matrix<weight, 1>; ///< The preferred type of input
    using output_one_t = etl::dyn_matrix<weight, 1>; ///< The type of output

    size_t S;

    /*!
     * \brief Initialize the dynamic layer
     */
    void init_layer(size_t S){
        cpp_assert(S > 1, "The shape must be bigger than 0");

        this->S = S;
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        return "Shape1D(dyn)";
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        return "Shape1D(dyn)";
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {S};
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    size_t input_size() const  {
        return S;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    size_t output_size() const {
        return S;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void forward_batch(Output& output, const Input& input) const {
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

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<dyn_shape_1d_layer_impl<Desc>> {
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
    static constexpr bool is_dynamic    = true; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dyn_lcn_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_shape_1d_layer_impl<Desc>, L> {
    using layer_t = dyn_shape_1d_layer_impl<Desc>;
    using weight  = typename DBN::weight; ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    using inputs_t = etl::dyn_matrix<weight, 2>;

    inputs_t input;  ///< A batch of input
    inputs_t output; ///< A batch of output
    inputs_t errors; ///< A batch of errors

    sgd_context(const layer_t& layer) : input(batch_size, layer.S), output(batch_size, layer.S), errors(batch_size, layer.S){}
};

} //end of dll namespace
