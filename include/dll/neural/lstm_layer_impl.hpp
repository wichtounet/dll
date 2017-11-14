//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/base_lstm_layer.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Standard dense layer of neural network.
 */
template <typename Desc>
struct lstm_layer_impl final : base_lstm_layer<lstm_layer_impl<Desc>, Desc> {
    using desc        = Desc;                             ///< The descriptor of the layer
    using weight      = typename desc::weight;            ///< The data type for this layer
    using this_type   = lstm_layer_impl<desc>;            ///< The type of this layer
    using base_type   = base_lstm_layer<this_type, desc>; ///< The base type
    using layer_t     = this_type;                        ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;       ///< The dynamic version of this layer

    static constexpr size_t time_steps      = desc::time_steps;               ///< The number of time steps
    static constexpr size_t sequence_length = desc::sequence_length;          ///< The length of the sequences
    static constexpr size_t hidden_units    = desc::hidden_units;             ///< The number of hidden units
    static constexpr size_t Z               = sequence_length + hidden_units; ///< The size of the internal input

    static constexpr size_t bptt_steps = desc::Truncate == 0 ? time_steps : desc::Truncate; ///< The number of bptt steps

    static constexpr auto activation_function = desc::activation_function; ///< The layer's activation function

    using w_initializer = typename desc::w_initializer; ///< The initializer for the W weights
    using u_initializer = typename desc::u_initializer; ///< The initializer for the U weights

    using input_one_t  = etl::fast_dyn_matrix<weight, time_steps, sequence_length>; ///< The type of one input
    using output_one_t = etl::fast_dyn_matrix<weight, time_steps, hidden_units>;    ///< The type of one output
    using input_t      = std::vector<input_one_t>;                                  ///< The type of the input
    using output_t     = std::vector<output_one_t>;                                 ///< The type of the output

    using w_type = etl::fast_matrix<weight, hidden_units, hidden_units>;    ///< The type of the W weights
    using u_type = etl::fast_matrix<weight, sequence_length, hidden_units>; ///< The type of the U weights
    using b_type = etl::fast_matrix<weight, hidden_units>;                  ///< The type of the biases

    //Weights and biases
    w_type w_i; ///< Weights W of the input gate
    u_type u_i; ///< Weights U of the input gate
    b_type b_i; ///< Biases of the input gate
    w_type w_g; ///< Weights W of the input modulation gate
    u_type u_g; ///< Weights U of the input modulation gate
    b_type b_g; ///< Biases of the input modulation gate
    w_type w_f; ///< Weights W of the forget gate
    u_type u_f; ///< Weights U of the forget gate
    b_type b_f; ///< Biases of the forget gate
    w_type w_o; ///< Weights W of the output gate
    u_type u_o; ///< Weights U of the output gate
    b_type b_o; ///< Biases of the output gate

    //Backup Weights and biases
    std::unique_ptr<w_type> bak_w_i; ///< Backup Weights W of the input gate
    std::unique_ptr<u_type> bak_u_i; ///< Backup Weights U of the input gate
    std::unique_ptr<b_type> bak_b_i; ///< Backup Biases of the input gate
    std::unique_ptr<w_type> bak_w_g; ///< Backup Weights W of the input modulation gate
    std::unique_ptr<u_type> bak_u_g; ///< Backup Weights U of the input modulation gate
    std::unique_ptr<b_type> bak_b_g; ///< Backup Biases of the input modulation gate
    std::unique_ptr<w_type> bak_w_f; ///< Backup Weights W of the forget gate
    std::unique_ptr<u_type> bak_u_f; ///< Backup Weights U of the forget gate
    std::unique_ptr<b_type> bak_b_f; ///< Backup Biases of the forget gate
    std::unique_ptr<w_type> bak_w_o; ///< Backup Weights W of the output gate
    std::unique_ptr<u_type> bak_u_o; ///< Backup Weights U of the output gate
    std::unique_ptr<b_type> bak_b_o; ///< Backup Biases of the output gate

    /*!
     * \brief Initialize a recurrent layer with basic weights.
     *
     * The weights are initialized from a normal distribution of
     * zero-mean and unit variance.
     */
    lstm_layer_impl() : base_type() {
        w_initializer::initialize(w_i, sequence_length, hidden_units);
        w_initializer::initialize(w_g, sequence_length, hidden_units);
        w_initializer::initialize(w_f, sequence_length, hidden_units);
        w_initializer::initialize(w_o, sequence_length, hidden_units);

        u_initializer::initialize(u_i, hidden_units, hidden_units);
        u_initializer::initialize(u_g, hidden_units, hidden_units);
        u_initializer::initialize(u_f, hidden_units, hidden_units);
        u_initializer::initialize(u_o, hidden_units, hidden_units);

        //TODO Initializer for the biases
        b_i = 0;
        b_g = 0;
        b_f = 0;
        b_o = 0;
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
        return 4 * hidden_units * hidden_units + 4 * hidden_units * sequence_length;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    static std::string to_short_string(std::string pre = "") {
        cpp_unused(pre);

        char buffer[512];

        if /*constexpr*/ (activation_function == function::IDENTITY) {
            snprintf(buffer, 512, "LSTM: %lux%lu -> %lux%lu", time_steps, sequence_length, time_steps, hidden_units);
        } else {
            snprintf(buffer, 512, "LSTM: %lux%lu -> %s -> %lux%lu", time_steps, sequence_length, to_string(activation_function).c_str(), time_steps, hidden_units);
        }

        return {buffer};
    }

    /*!
     * \brief Apply the layer to the given batch of input.
     *
     * \param x A batch of input
     * \param output A batch of output that will be filled
     */
    template <typename H, typename V>
    void forward_batch(H&& output, const V& x) const {
        dll::auto_timer timer("lstm:forward_batch");

        const auto Batch = etl::dim<0>(x);

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        etl::dyn_matrix<float, 3> x_t(time_steps, Batch, sequence_length);
        etl::dyn_matrix<float, 3> g_t(time_steps, Batch, hidden_units);
        etl::dyn_matrix<float, 3> i_t(time_steps, Batch, hidden_units);
        etl::dyn_matrix<float, 3> f_t(time_steps, Batch, hidden_units);
        etl::dyn_matrix<float, 3> o_t(time_steps, Batch, hidden_units);
        etl::dyn_matrix<float, 3> s_t(time_steps, Batch, hidden_units);
        etl::dyn_matrix<float, 3> h_t(time_steps, Batch, hidden_units);

        // 1. Rearrange input

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                x_t(t)(b) = x(b)(t);
            }
        }

        // 2. Forward propagation through time

        // t == 0

        g_t(0) =    etl::tanh(bias_add_2d(x_t(0) * (u_g), b_g));
        i_t(0) = etl::sigmoid(bias_add_2d(x_t(0) * (u_i), b_i));
        f_t(0) = etl::sigmoid(bias_add_2d(x_t(0) * (u_f), b_f));
        o_t(0) = etl::sigmoid(bias_add_2d(x_t(0) * (u_o), b_o));

        s_t(0) = g_t(0) >> i_t(0);
        h_t(0) = f_activate<activation_function>(s_t(0)) >> o_t(0);

        for (size_t t = 1; t < time_steps; ++t) {
            g_t(t) =    etl::tanh(bias_add_2d(x_t(t) * u_g + h_t(t - 1) * w_g, b_g));
            i_t(t) = etl::sigmoid(bias_add_2d(x_t(t) * u_i + h_t(t - 1) * w_i, b_i));
            f_t(t) = etl::sigmoid(bias_add_2d(x_t(t) * u_f + h_t(t - 1) * w_f, b_f));
            o_t(t) = etl::sigmoid(bias_add_2d(x_t(t) * u_o + h_t(t - 1) * w_o, b_o));

            s_t(t) = (g_t(t) >> i_t(t)) + (s_t(t - 1) >> f_t(t));
            h_t(t) = f_activate<activation_function>(s_t(t)) >> o_t(t);
        }

        // 3. Rearrange the output

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                output(b)(t) = h_t(t)(b);
            }
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
    template <typename DLayer>
    static void dyn_init(DLayer& dyn) {
        cpp_unused(dyn);
        //TODO dyn.init_layer(time_steps, sequence_length, hidden_units);
    }

    /*!
     * \brief Adapt the errors, called before backpropagation of the errors.
     *
     * This must be used by layers that have both an activation fnction and a non-linearity.
     *
     * \param context the training context
     */
    template <typename C>
    void adapt_errors(C& context) const {
        // Nothing to do here (done in BPTT)
        cpp_unused(context);
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template <typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        dll::auto_timer timer("lstm:backward_batch");

        cpp_unused(output);
        cpp_unused(context);

        //TODO base_type::backward_batch_impl(output, context, w, time_steps, sequence_length, hidden_units, bptt_steps);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template <typename C>
    void compute_gradients(C& context) const {
        const size_t Batch = etl::dim<0>(context.errors);

        auto& x = context.input;
        auto& h = context.output;

        etl::dyn_matrix<float, 3> x_t(time_steps, Batch, sequence_length);
        etl::dyn_matrix<float, 3> h_t(time_steps, Batch, hidden_units);
        etl::dyn_matrix<float, 3> o_t(time_steps, Batch, hidden_units);

        // 1. Rearrange x/h

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                x_t(t)(b) = x(b)(t);
            }
        }

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                h_t(t)(b) = h(b)(t);
            }
        }

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                o_t(t)(b) = context.errors(b)(t);
            }
        }

        auto& w_i_grad = std::get<0>(context.up.context)->grad;
        auto& u_i_grad = std::get<1>(context.up.context)->grad;
        auto& b_i_grad = std::get<1>(context.up.context)->grad;
        auto& w_g_grad = std::get<0>(context.up.context)->grad;
        auto& u_g_grad = std::get<1>(context.up.context)->grad;
        auto& b_g_grad = std::get<1>(context.up.context)->grad;
        auto& w_f_grad = std::get<0>(context.up.context)->grad;
        auto& u_f_grad = std::get<1>(context.up.context)->grad;
        auto& b_f_grad = std::get<1>(context.up.context)->grad;
        auto& w_o_grad = std::get<0>(context.up.context)->grad;
        auto& u_o_grad = std::get<1>(context.up.context)->grad;
        auto& b_o_grad = std::get<1>(context.up.context)->grad;

        w_i_grad = 0;
        u_i_grad = 0;
        b_i_grad = 0;
        w_g_grad = 0;
        u_g_grad = 0;
        b_g_grad = 0;
        w_f_grad = 0;
        u_f_grad = 0;
        b_f_grad = 0;
        w_o_grad = 0;
        u_o_grad = 0;
        b_o_grad = 0;

        size_t t = time_steps - 1;

        do {
            //TODO

            size_t bptt_step = t;

            const size_t last_step = std::max(int(time_steps) - int(bptt_steps), 0);

            do {
                //TODO

                --bptt_step;
            } while (bptt_step > last_step);

            // bptt_step = 0

            //TODO

            --t;

            // If only the last time step is used, no need to use the other errors
            if /*constexpr*/ (desc::parameters::template contains<last_only>()) {
                break;
            }
        } while (t != 0);
    }
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const size_t lstm_layer_impl<Desc>::time_steps;

template <typename Desc>
const size_t lstm_layer_impl<Desc>::sequence_length;

template <typename Desc>
const size_t lstm_layer_impl<Desc>::hidden_units;

// Declare the traits for the Layer

template <typename Desc>
struct layer_base_traits<lstm_layer_impl<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief specialization of sgd_context for lstm_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, lstm_layer_impl<Desc>, L> {
    using layer_t = lstm_layer_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr size_t time_steps      = layer_t::time_steps;      ///< The number of time steps
    static constexpr size_t sequence_length = layer_t::sequence_length; ///< The length of the sequences
    static constexpr size_t hidden_units    = layer_t::hidden_units;    ///< The number of hidden units

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, time_steps, sequence_length> input;
    etl::fast_matrix<weight, batch_size, time_steps, hidden_units> output;
    etl::fast_matrix<weight, batch_size, time_steps, hidden_units> errors;

    sgd_context(const lstm_layer_impl<Desc>& /* layer */)
            : output(0.0), errors(0.0) {}
};

} //end of dll namespace
