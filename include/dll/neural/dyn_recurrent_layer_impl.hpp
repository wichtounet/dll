//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/recurrent_neural_layer.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Standard dense layer of neural network.
 */
template <typename Desc>
struct dyn_recurrent_layer_impl final : recurrent_neural_layer<dyn_recurrent_layer_impl<Desc>, Desc> {
    using desc        = Desc;                                    ///< The descriptor of the layer
    using weight      = typename desc::weight;                   ///< The data type for this layer
    using this_type   = dyn_recurrent_layer_impl<desc>;          ///< The type of this layer
    using base_type   = recurrent_neural_layer<this_type, desc>; ///< The base type
    using layer_t     = this_type;                               ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;              ///< The dynamic version of this layer

    static constexpr auto activation_function = desc::activation_function; ///< The layer's activation function

    using w_initializer = typename desc::w_initializer; ///< The initializer for the W weights
    using u_initializer = typename desc::u_initializer; ///< The initializer for the U weights

    using input_one_t  = etl::dyn_matrix<weight, 2>; ///< The type of one input
    using output_one_t = etl::dyn_matrix<weight, 2>; ///< The type of one output
    using input_t      = std::vector<input_one_t>;   ///< The type of the input
    using output_t     = std::vector<output_one_t>;  ///< The type of the output

    using w_type = etl::dyn_matrix<weight, 2>; ///< The type of the W weights
    using u_type = etl::dyn_matrix<weight, 2>; ///< The type of the U weights

    //Weights and biases
    w_type w; ///< Weights W
    u_type u; ///< Weights U

    //Backup Weights and biases
    std::unique_ptr<w_type> bak_w; ///< Backup Weights W
    std::unique_ptr<u_type> bak_u; ///< Backup Weights U

    size_t time_steps;      ///< The number of time steps
    size_t sequence_length; ///< The length of the sequences
    size_t hidden_units;    ///< The number of hidden units
    size_t bptt_steps;      ///< The number of BPTT steps

    /*!
     * \brief Initialize a recurrent layer with basic weights.
     *
     * The weights are initialized from a normal distribution of
     * zero-mean and unit variance.
     */
    dyn_recurrent_layer_impl() : base_type() {}

    /*!
     * \brief Initialize the dynamic layer
     */
    void init_layer(size_t time_steps, size_t sequence_length, size_t hidden_units) {
        this->time_steps      = time_steps;
        this->sequence_length = sequence_length;
        this->hidden_units    = hidden_units;

        this->bptt_steps = desc::Truncate == 0 ? time_steps : desc::Truncate;

        w = etl::dyn_matrix<weight, 2>(hidden_units, hidden_units);
        u = etl::dyn_matrix<weight, 2>(hidden_units, sequence_length);

        w_initializer::initialize(w, hidden_units, hidden_units);
        u_initializer::initialize(u, hidden_units, hidden_units);
    }

    /*!
     * \brief Returns the input size of this layer
     */
    size_t input_size() const noexcept {
        return time_steps * sequence_length;
    }

    /*!
     * \brief Returns the output size of this layer
     */
    size_t output_size() const noexcept {
        return time_steps * hidden_units;
    }

    /*!
     * \brief Returns the number of parameters of this layer
     */
    size_t parameters() const noexcept {
        return hidden_units * hidden_units + hidden_units * sequence_length;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_short_string(std::string pre = "") const {
        cpp_unused(pre);

        char buffer[512];

        if /*constexpr*/ (activation_function == function::IDENTITY) {
            snprintf(buffer, 512, "RNN(dyn): %lux%lu -> %lux%lu", time_steps, sequence_length, time_steps, hidden_units);
        } else {
            snprintf(buffer, 512, "RNN(dyn): %lux%lu -> %s -> %lux%lu", time_steps, sequence_length, to_string(activation_function).c_str(), time_steps, hidden_units);
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
        dll::auto_timer timer("recurrent:forward_batch");

        const auto Batch = etl::dim<0>(x);

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        etl::dyn_matrix<float, 3> x_t(time_steps, etl::dim<0>(x), sequence_length);
        etl::dyn_matrix<float, 3> s_t(time_steps, etl::dim<0>(x), hidden_units);

        // 1. Rearrange input

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                x_t(t)(b) = x(b)(t);
            }
        }

        // 2. Forward propagation through time

        // t == 0

        s_t(0) = f_activate<activation_function>(x_t(0) * trans(u));

        for (size_t t = 1; t < time_steps; ++t) {
            s_t(t) = f_activate<activation_function>(x_t(t) * trans(u) + s_t(t - 1) * trans(w));
        }

        // 3. Rearrange the output

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                output(b)(t) = s_t(t)(b);
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
        return output_one_t(time_steps, hidden_units);
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    output_t prepare_output(size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(size_t i = 0; i < samples; ++i){
            output.emplace_back(time_steps, hidden_units);
        }
        return output;
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DLayer>
    static void dyn_init(DLayer& dyn){
        cpp_unused(dyn);
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
        // Nothing to do here (done in BPTT)
        cpp_unused(context);
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        dll::auto_timer timer("recurrent:backward_batch");

        const size_t Batch = etl::dim<0>(context.errors);

        etl::dyn_matrix<float, 3> output_t(time_steps, Batch, sequence_length);
        etl::dyn_matrix<float, 3> s_t(time_steps, Batch, hidden_units);
        etl::dyn_matrix<float, 3> o_t(time_steps, Batch, hidden_units);

        // 1. Rearrange o/s

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                s_t(t)(b) = context.output(b)(t);
            }
        }

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                o_t(t)(b) = context.errors(b)(t);
            }
        }

        // 2. Backpropagation through time

        size_t t = time_steps - 1;

        do {
            output_t(t) = etl::force_temporary(o_t(t) >> f_derivative<activation_function>(s_t(t)));

            size_t bptt_step = t;

            const size_t last_step = std::max(int(time_steps) - int(bptt_steps), 0);

            do {
                output_t(t) = (output_t(t) * w) >> f_derivative<activation_function>(s_t(bptt_step - 1));

                --bptt_step;
            } while (bptt_step > last_step);

            // bptt_step = 0

            --t;

            // If only the last time step is used, no need to use the other errors
            if /*constexpr*/ (desc::parameters::template contains<last_only>()){
                break;
            }
        } while(t != 0);

        // 3. Rearrange for the output

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                output(b)(t) = output_t(t)(b);
            }
        }
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        dll::auto_timer timer("recurrent:compute_gradients");

        const size_t Batch = etl::dim<0>(context.errors);

        auto& x = context.input;
        auto& s = context.output;

        etl::dyn_matrix<float, 3> x_t(time_steps, Batch, sequence_length);
        etl::dyn_matrix<float, 3> s_t(time_steps, Batch, hidden_units);
        etl::dyn_matrix<float, 3> o_t(time_steps, Batch, hidden_units);

        // 1. Rearrange x/s

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                x_t(t)(b) = x(b)(t);
            }
        }

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                s_t(t)(b) = s(b)(t);
            }
        }

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                o_t(t)(b) = context.errors(b)(t);
            }
        }

        auto& w_grad = std::get<0>(context.up.context)->grad;
        auto& u_grad = std::get<1>(context.up.context)->grad;

        w_grad = 0;
        u_grad = 0;

        size_t t = time_steps - 1;

        do {
            auto delta_t = etl::force_temporary(o_t(t) >> f_derivative<activation_function>(s_t(t)));

            size_t bptt_step = t;

            const size_t last_step = std::max(int(time_steps) - int(bptt_steps), 0);

            do {
                w_grad += etl::batch_outer(delta_t, s_t(bptt_step - 1));
                u_grad += etl::batch_outer(delta_t, x_t(bptt_step));

                delta_t = (delta_t * w) >> f_derivative<activation_function>(s_t(bptt_step - 1));

                --bptt_step;
            } while (bptt_step > last_step);

            // bptt_step = 0

            u_grad += etl::batch_outer(delta_t, x_t(0));

            --t;

            // If only the last time step is used, no need to use the other errors
            if /*constexpr*/ (desc::parameters::template contains<last_only>()){
                break;
            }
        } while(t != 0);
    }
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<dyn_recurrent_layer_impl<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_dynamic    = true;  ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief specialization of sgd_context for dyn_recurrent_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_recurrent_layer_impl<Desc>, L> {
    using layer_t = dyn_recurrent_layer_impl<Desc>; ///< The layer
    using weight  = typename layer_t::weight;       ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::dyn_matrix<weight, 3> input;
    etl::dyn_matrix<weight, 3> output;
    etl::dyn_matrix<weight, 3> errors;

    sgd_context(const dyn_recurrent_layer_impl<Desc>& layer)
            : input(batch_size, layer.time_steps, layer.sequence_length), output(batch_size, layer.time_steps, layer.hidden_units, 0.0), errors(batch_size, layer.time_steps, layer.hidden_units, 0.0) {}
};

} //end of dll namespace
