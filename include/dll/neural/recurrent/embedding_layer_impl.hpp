//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/neural_layer_no_bias.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Standard embedding layer of neural network.
 */
template <typename Desc>
struct embedding_layer_impl final : neural_layer_no_bias<embedding_layer_impl<Desc>, Desc> {
    using desc      = Desc;                                  ///< The descriptor of the layer
    using weight    = typename desc::weight;                 ///< The data type of the layer
    using this_type = embedding_layer_impl<desc>;            ///< The type of this layer
    using base_type = neural_layer_no_bias<this_type, desc>; ///< The base type of the layer
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    static inline constexpr size_t V = desc::V; ///< The vocabulary size
    static inline constexpr size_t I = desc::I; ///< The input size
    static inline constexpr size_t K = desc::K; ///< The embedding size

    using w_initializer = typename desc::w_initializer; ///< The initializer for the weights

    using input_one_t  = etl::fast_dyn_matrix<weight, I>;    ///< The type of one input
    using output_one_t = etl::fast_dyn_matrix<weight, I, K>; ///< The type of one output
    using input_t      = std::vector<input_one_t>;           ///< The type of the input
    using output_t     = std::vector<output_one_t>;          ///< The type of the output

    using w_type = etl::fast_matrix<weight, V, K>; ///< The type of the weights

    //Weights and biases
    w_type w; ///< Weights

    //Backup weights and biases
    std::unique_ptr<w_type> bak_w; ///< Backup Weights

    /*!
     * \brief Initialize a embedding layer with basic weights.
     */
    embedding_layer_impl() : base_type() {
        w_initializer::initialize(w, input_size(), output_size());
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    static constexpr size_t input_size() noexcept {
        return I;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    static constexpr size_t output_size() noexcept {
        return I * K;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    static constexpr size_t parameters() noexcept {
        return V * K;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        return "Embedding";
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        char buffer[256];
        snprintf(buffer, 256, "Embedding: %lu -> (%lux%lu) -> %lu", I, V, K, K);
        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {K};
    }

    /*!
     * \brief Apply the layer to the given batch of input.
     *
     * \param input A batch of input
     * \param output A batch of output that will be filled
     */
    template <typename H1, typename V>
    void forward_batch(H1&& output, const V& v) const {
        dll::auto_timer timer("embedding:forward_batch");

        output = batch_embedding_lookup(v, w);
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     */
    template <typename Input>
    output_one_t prepare_one_output() const {
        return {};
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     */
    template <typename Input>
    static output_t prepare_output(size_t samples) {
        return output_t{samples};
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(V, I, K);
    }

    /*!
     * \brief Adapt the errors, called before backpropagation of the errors.
     *
     * This must be used by layers that have both an activation fnction and a non-linearity.
     *
     * \param context the training context
     */
    template<typename C>
    void adapt_errors([[maybe_unused]] C& context) const {
        // Nothing to adapt
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        dll::auto_timer timer("embedding:compute_gradients");

        std::get<0>(context.up.context)->grad = batch_embedding_gradients(context.input, context.errors, w);;
    }
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<embedding_layer_impl<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of the sgd_context for embedding_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, embedding_layer_impl<Desc>, L> {
    using layer_t = embedding_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr size_t V   = layer_t::V;
    static constexpr size_t I   = layer_t::I;
    static constexpr size_t K   = layer_t::K;

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, I> input;
    etl::fast_matrix<weight, batch_size, I, K> output;
    etl::fast_matrix<weight, batch_size, I, K> errors;

    sgd_context(const embedding_layer_impl<Desc>& /* layer */)
            : output(0.0), errors(0.0) {}
};

} //end of dll namespace
