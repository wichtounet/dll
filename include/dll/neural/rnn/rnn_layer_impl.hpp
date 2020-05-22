//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/base_rnn_layer.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Standard dense layer of neural network.
 */
template <typename Desc>
struct rnn_layer_impl final : base_rnn_layer<rnn_layer_impl<Desc>, Desc> {
    using desc        = Desc;                            ///< The descriptor of the layer
    using weight      = typename desc::weight;           ///< The data type for this layer
    using this_type   = rnn_layer_impl<desc>;      ///< The type of this layer
    using base_type   = base_rnn_layer<this_type, desc>; ///< The base type
    using layer_t     = this_type;                       ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;      ///< The dynamic version of this layer

    static constexpr size_t time_steps      = desc::time_steps;      ///< The number of time steps
    static constexpr size_t sequence_length = desc::sequence_length; ///< The length of the sequences
    static constexpr size_t hidden_units    = desc::hidden_units;    ///< The number of hidden units

    static constexpr size_t bptt_steps = desc::Truncate == 0 ? time_steps : desc::Truncate; ///< The number of bptt steps

    static constexpr auto activation_function = desc::activation_function; ///< The layer's activation function

    using w_initializer = typename desc::w_initializer; ///< The initializer for the W weights
    using u_initializer = typename desc::u_initializer; ///< The initializer for the U weights
    using b_initializer = typename desc::b_initializer; ///< The initializer for the biases

    using input_one_t  = etl::fast_dyn_matrix<weight, time_steps, sequence_length>; ///< The type of one input
    using output_one_t = etl::fast_dyn_matrix<weight, time_steps, hidden_units>;    ///< The type of one output
    using input_t      = std::vector<input_one_t>;                                  ///< The type of the input
    using output_t     = std::vector<output_one_t>;                                 ///< The type of the output

    using w_type = etl::fast_matrix<weight, hidden_units, hidden_units>;    ///< The type of the W weights
    using u_type = etl::fast_matrix<weight, sequence_length, hidden_units>; ///< The type of the U weights
    using b_type = etl::fast_matrix<weight, hidden_units>;                  ///< The type of the U weights

    //Weights and biases
    w_type w; ///< Weights W
    u_type u; ///< Weights U
    b_type b; ///< Biases b

    //Backup Weights and biases
    std::unique_ptr<w_type> bak_w; ///< Backup Weights W
    std::unique_ptr<u_type> bak_u; ///< Backup Weights U
    std::unique_ptr<b_type> bak_b; ///< Backup Weights U

    /*!
     * \brief Initialize a recurrent layer with basic weights.
     *
     * The weights are initialized from a normal distribution of
     * zero-mean and unit variance.
     */
    rnn_layer_impl()
            : base_type() {
        w_initializer::initialize(w, hidden_units, hidden_units);
        u_initializer::initialize(u, hidden_units, hidden_units);
        b_initializer::initialize(b, sequence_length, hidden_units);
    }

    /*!
     * \brief Returns the input size of this layer
     */
    static constexpr size_t input_size() noexcept {
        return time_steps * sequence_length;
    }

    /*!
     * \brief Returns the output size of this layer
     */
    static constexpr size_t output_size() noexcept {
        return time_steps * hidden_units;
    }

    /*!
     * \brief Returns the number of parameters of this layer
     */
    static constexpr size_t parameters() noexcept {
        return hidden_units * hidden_units + hidden_units * sequence_length + hidden_units;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        if constexpr (activation_function == function::IDENTITY) {
            return "RNN";
        } else {
            char buffer[512];
            snprintf(buffer, 512, "RNN(%s)", to_string(activation_function).c_str());
            return {buffer};
        }
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        char buffer[512];

        if constexpr (activation_function == function::IDENTITY) {
            snprintf(buffer, 512, "RNN: %lux%lu -> %lux%lu", time_steps, sequence_length, time_steps, hidden_units);
        } else {
            snprintf(buffer, 512, "RNN: %lux%lu -> %s -> %lux%lu", time_steps, sequence_length, to_string(activation_function).c_str(), time_steps, hidden_units);
        }

        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {time_steps, hidden_units};
    }

    /*!
     * \brief Apply the layer to the given batch of input.
     *
     * \param x A batch of input
     * \param output A batch of output that will be filled
     */
    template <typename H, typename V>
    void forward_batch(H&& output, const V& x) const {
        dll::auto_timer timer("rnn:forward_batch");

        cpp_assert(etl::dim<0>(output) == etl::dim<0>(x), "The number of samples must be consistent");

        base_type::forward_batch_impl(output, x, w, u, b, time_steps, sequence_length, hidden_units);
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
    template <typename DLayer>
    static void dyn_init(DLayer& dyn) {
        dyn.init_layer(time_steps, sequence_length, hidden_units);
    }

    /*!
     * \brief Adapt the errors, called before backpropagation of the errors.
     *
     * This must be used by layers that have both an activation fnction and a non-linearity.
     *
     * \param context the training context
     */
    template <typename C>
    void adapt_errors([[maybe_unused]] C& context) const {
        // Nothing to do here (done in BPTT)
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template <typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        dll::auto_timer timer("rnn:backward_batch");

        base_type::backward_batch_impl(output, context, w, u, time_steps, sequence_length, hidden_units, bptt_steps);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template <typename C>
    void compute_gradients(C& context) const {
        dll::auto_timer timer("rnn:compute_gradients");

        base_type::compute_gradients_impl(context, w, u, time_steps, sequence_length, hidden_units, bptt_steps);
    }
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const size_t rnn_layer_impl<Desc>::time_steps;

template <typename Desc>
const size_t rnn_layer_impl<Desc>::sequence_length;

template <typename Desc>
const size_t rnn_layer_impl<Desc>::hidden_units;

// Declare the traits for the Layer

template <typename Desc>
struct layer_base_traits<rnn_layer_impl<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = true; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief specialization of sgd_context for rnn_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, rnn_layer_impl<Desc>, L> {
    using layer_t = rnn_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr size_t time_steps      = layer_t::time_steps;      ///< The number of time steps
    static constexpr size_t sequence_length = layer_t::sequence_length; ///< The length of the sequences
    static constexpr size_t hidden_units    = layer_t::hidden_units;    ///< The number of hidden units

    static constexpr size_t layer    = L;               ///< The index of the layer
    static constexpr auto batch_size = DBN::batch_size; ///< The batch size of the network

    etl::fast_matrix<weight, batch_size, time_steps, sequence_length> input;
    etl::fast_matrix<weight, batch_size, time_steps, hidden_units> output;
    etl::fast_matrix<weight, batch_size, time_steps, hidden_units> errors;

    sgd_context(const rnn_layer_impl<Desc>& /* layer */)
            : output(0.0), errors(0.0) {}
};

} //end of dll namespace
