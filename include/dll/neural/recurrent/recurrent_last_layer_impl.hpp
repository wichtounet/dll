//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Standard dense layer of neural network.
 */
template <typename Desc>
struct recurrent_last_layer_impl final : layer<recurrent_last_layer_impl<Desc>> {
    using desc        = Desc;                            ///< The descriptor of the layer
    using weight      = typename desc::weight;           ///< The data type for this layer
    using this_type   = recurrent_last_layer_impl<desc>; ///< The type of this layer
    using base_type   = layer<this_type>;                ///< The base type
    using layer_t     = this_type;                       ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;      ///< The dynamic version of this layer

    static constexpr size_t time_steps   = desc::time_steps;   ///< The number of time steps
    static constexpr size_t hidden_units = desc::hidden_units; ///< The number of hidden units

    using input_one_t  = etl::fast_dyn_matrix<weight, time_steps, hidden_units>; ///< The type of one input
    using output_one_t = etl::fast_dyn_matrix<weight, hidden_units>;             ///< The type of one output
    using input_t      = std::vector<input_one_t>;                               ///< The type of the input
    using output_t     = std::vector<output_one_t>;                              ///< The type of the output

    /*!
     * \brief Returns the input size of this layer
     */
    static constexpr size_t input_size() noexcept {
        return time_steps * hidden_units;
    }

    /*!
     * \brief Returns the output size of this layer
     */
    static constexpr size_t output_size() noexcept {
        return hidden_units;
    }

    /*!
     * \brief Returns the number of parameters of this layer
     */
    static constexpr size_t parameters() noexcept {
        return 0;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        char buffer[512];
        snprintf(buffer, 512, "RNN(last)");
        return {buffer};
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        char buffer[512];
        snprintf(buffer, 512, "RNN(last): %lux%lu -> %lu", time_steps, hidden_units, hidden_units);
        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {hidden_units};
    }

    /*!
     * \brief Apply the layer to the given batch of input.
     *
     * \param input A batch of input
     * \param output A batch of output that will be filled
     */
    template <typename H, typename V>
    void forward_batch(H&& output, const V& input) const {
        dll::auto_timer timer("recurrent_last:forward_batch");

        const auto Batch = etl::dim<0>(input);

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        for(size_t b = 0; b < Batch; ++b){
            output(b) = input(b)(time_steps - 1);
        }
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    output_one_t prepare_one_output() const {
        return {};
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
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
    template<typename DLayer>
    static void dyn_init(DLayer& dyn){
        dyn.init_layer(time_steps, hidden_units);
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
        // Nothing to do here
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        dll::auto_timer timer("recurrent_last:backward_batch");

        const auto Batch = etl::dim<0>(output);

        output = 0;

        for(size_t b = 0; b < Batch; ++b){
            output(b)(time_steps - 1) = context.errors(b);
        }
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients([[maybe_unused]] C& context) const {
        // Nothing to do here
    }
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const size_t recurrent_last_layer_impl<Desc>::time_steps;

template <typename Desc>
const size_t recurrent_last_layer_impl<Desc>::hidden_units;

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<recurrent_last_layer_impl<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
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
 * \brief specialization of sgd_context for recurrent_last_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, recurrent_last_layer_impl<Desc>, L> {
    using layer_t = recurrent_last_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr size_t time_steps      = layer_t::time_steps;      ///< The number of time steps
    static constexpr size_t hidden_units    = layer_t::hidden_units;    ///< The number of hidden units

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, time_steps, hidden_units> input;
    etl::fast_matrix<weight, batch_size, hidden_units> output;
    etl::fast_matrix<weight, batch_size, hidden_units> errors;

    sgd_context(const recurrent_last_layer_impl<Desc>& /* layer */)
            : output(0.0), errors(0.0) {}
};

} //end of dll namespace
