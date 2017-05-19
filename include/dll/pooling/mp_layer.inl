//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "pooling_layer.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Standard max pooling layer (2D pooling)
 */
template <typename Desc>
struct mp_layer_2d final : pooling_layer_2d<mp_layer_2d<Desc>, Desc> {
    using desc   = Desc;                                      ///< The layer descriptor
    using weight = typename desc::weight;                     ///< The layer weight type
    using base   = pooling_layer_2d<mp_layer_2d<Desc>, desc>; ///< The layer base type

    mp_layer_2d() = default;

    /*!
     * \brief Get a string representation of the layer
     */
    static std::string to_short_string() {
        char buffer[1024];
        snprintf(buffer, 1024, "MP(2D): %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu",
                 base::I1, base::I2, base::I3, base::C1, base::C2, base::O1, base::O2, base::O3);
        return {buffer};
    }

    using input_one_t  = typename base::input_one_t;  ///< The type of one input
    using output_one_t = typename base::output_one_t; ///< The type of one output
    using input_t      = typename base::input_t;      ///< The type of many input
    using output_t     = typename base::output_t;     ///< The type of many output

    using base::activate_hidden;

    /*!
     * \brief Forward activation of the layer for one sample
     * \param h The output matrix
     * \param v The input matrix
     */
    template <typename Input>
    static void activate_hidden(output_one_t& h, const Input& v) {
        dll::auto_timer timer("mp:activate_hidden");

        h = etl::max_pool_2d<base::C1, base::C2>(v);
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \return A batch of output corresponding to the activated input
     */
    template <typename V, cpp_enable_if(etl::decay_traits<V>::is_fast)>
    auto batch_activate_hidden(const V& v) const {
        static constexpr auto Batch = etl::decay_traits<V>::template dim<0>();
        etl::fast_dyn_matrix<weight, Batch, base::O1, base::O2, base::O3> output;
        batch_activate_hidden(output, v);
        return output;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \return A batch of output corresponding to the activated input
     */
    template <typename V, cpp_enable_if(!etl::decay_traits<V>::is_fast)>
    auto batch_activate_hidden(const V& v) const {
        const auto Batch = etl::dim<0>(v);
        etl::dyn_matrix<weight, 4> output(Batch, base::O1, base::O2, base::O3);
        batch_activate_hidden(output, v);
        return output;
    }

    /*!
     * \brief Forward activation of the layer for one batch of sample
     * \param output The output matrix
     * \param input The input matrix
     */
    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        dll::auto_timer timer("mp:batch_activate_hidden");

        output = etl::max_pool_2d<base::C1, base::C2>(input);
    }

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
        dll::auto_timer timer("mp:backward_batch");

        static constexpr size_t C1 = base::C1;
        static constexpr size_t C2 = base::C2;

        output = etl::max_pool_upsample_2d<C1, C2>(context.input, context.output, context.errors);
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

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<mp_layer_2d<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = true;  ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = false; ///< Indicates if the layer is a patches layer
    static constexpr bool is_augment    = false; ///< Indicates if the layer is an augment layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for mp_layer_2d
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, mp_layer_2d<Desc>, L> {
    using layer_t = mp_layer_2d<Desc>;
    using weight  = typename layer_t::weight;

    static constexpr size_t I1 = layer_t::I1;
    static constexpr size_t I2 = layer_t::I2;
    static constexpr size_t I3 = layer_t::I3;

    static constexpr size_t O1 = layer_t::O1;
    static constexpr size_t O2 = layer_t::O2;
    static constexpr size_t O3 = layer_t::O3;

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, I1, I2, I3> input;
    etl::fast_matrix<weight, batch_size, O1, O2, O3> output;
    etl::fast_matrix<weight, batch_size, O1, O2, O3> errors;

    sgd_context(mp_layer_2d<Desc>& /*layer*/){}
};

/*!
 * \brief Standard max pooling layer (3D pooling)
 */
template <typename Desc>
struct mp_layer_3d final : pooling_layer_3d<mp_layer_3d<Desc>, Desc> {
    using desc   = Desc;                                      ///< The layer descriptor
    using weight = typename desc::weight;                     ///< The layer weight type
    using base   = pooling_layer_3d<mp_layer_3d<Desc>, desc>; ///< The layer base type

    mp_layer_3d() = default;

    /*!
     * \brief Get a string representation of the layer
     */
    static std::string to_short_string() {
        char buffer[1024];
        snprintf(buffer, 1024, "MP(3D): %lux%lux%lu -> (%lux%lux%lu) -> %lux%lux%lu",
                 base::I1, base::I2, base::I3, base::C1, base::C2, base::C3, base::O1, base::O2, base::O3);
        return {buffer};
    }

    using input_one_t  = typename base::input_one_t;  ///< The type of one input
    using output_one_t = typename base::output_one_t; ///< The type of one output
    using input_t      = typename base::input_t;      ///< The type of many input
    using output_t     = typename base::output_t;     ///< The type of many output

    using base::activate_hidden;

    /*!
     * \brief Forward activation of the layer for one sample
     * \param h The output matrix
     * \param v The input matrix
     */
    template <typename Input>
    static void activate_hidden(output_one_t& h, const Input& v) {
        dll::auto_timer timer("mp:activate_hidden");

        h = etl::max_pool_3d<base::C1, base::C2, base::C3>(v);
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \return A batch of output corresponding to the activated input
     */
    template <typename V, cpp_enable_if(etl::decay_traits<V>::is_fast)>
    auto batch_activate_hidden(const V& v) const {
        static constexpr auto Batch = etl::decay_traits<V>::template dim<0>();
        etl::fast_dyn_matrix<weight, Batch, base::O1, base::O2, base::O3> output;
        batch_activate_hidden(output, v);
        return output;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \return A batch of output corresponding to the activated input
     */
    template <typename V, cpp_enable_if(!etl::decay_traits<V>::is_fast)>
    auto batch_activate_hidden(const V& v) const {
        const auto Batch = etl::dim<0>(v);
        etl::dyn_matrix<weight, 4> output(Batch, base::O1, base::O2, base::O3);
        batch_activate_hidden(output, v);
        return output;
    }

    /*!
     * \brief Forward activation of the layer for one batch of sample
     * \param output The output matrix
     * \param input The input matrix
     */
    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        dll::auto_timer timer("mp:batch_activate_hidden");

        output = etl::max_pool_3d<base::C1, base::C2, base::C3>(input);
    }

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
        dll::auto_timer timer("mp:backward_batch");

        static constexpr size_t C1 = base::C1;
        static constexpr size_t C2 = base::C2;
        static constexpr size_t C3 = base::C3;

        output = etl::max_pool_upsample_3d<C1, C2, C3>(context.input, context.output, context.errors);
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

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<mp_layer_3d<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = true;  ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = false; ///< Indicates if the layer is a patches layer
    static constexpr bool is_augment    = false; ///< Indicates if the layer is an augment layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for mp_layer_3d
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, mp_layer_3d<Desc>, L> {
    using layer_t = mp_layer_3d<Desc>;
    using weight  = typename layer_t::weight;

    static constexpr size_t I1 = layer_t::I1;
    static constexpr size_t I2 = layer_t::I2;
    static constexpr size_t I3 = layer_t::I3;

    static constexpr size_t O1 = layer_t::O1;
    static constexpr size_t O2 = layer_t::O2;
    static constexpr size_t O3 = layer_t::O3;

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, I1, I2, I3> input;
    etl::fast_matrix<weight, batch_size, O1, O2, O3> output;
    etl::fast_matrix<weight, batch_size, O1, O2, O3> errors;

    sgd_context(mp_layer_3d<Desc>& /*layer*/){}
};

} //end of dll namespace
