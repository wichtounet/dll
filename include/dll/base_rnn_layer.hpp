//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <fstream>

#include "cpp_utils/assert.hpp" //Assertions
#include "cpp_utils/io.hpp"     // For binary writing

#include "etl/etl.hpp"

#include "layer.hpp"
#include "layer_traits.hpp"
#include "util/tmp.hpp"

namespace dll {

/*!
 * \brief Base class for RNN layers (fast / dynamic)
 */
template <typename Derived, typename Desc>
struct base_rnn_layer : layer<Derived> {
    using desc      = Desc;                            ///< The descriptor of the layer
    using derived_t = Derived;                         ///< The derived type (CRTP)
    using weight    = typename desc::weight;           ///< The data type for this layer
    using this_type = base_rnn_layer<derived_t, desc>; ///< The type of this layer
    using base_type = layer<Derived>;                  ///< The base type

    static constexpr auto activation_function = desc::activation_function; ///< The layer's activation function

    /*!
     * \brief Initialize the neural layer
     */
    base_rnn_layer()
            : base_type() {
        // Nothing to init here
    }

    base_rnn_layer(const base_rnn_layer& rhs) = delete;
    base_rnn_layer(base_rnn_layer&& rhs)      = delete;

    base_rnn_layer& operator=(const base_rnn_layer& rhs) = delete;
    base_rnn_layer& operator=(base_rnn_layer&& rhs) = delete;

    mutable etl::dyn_matrix<float, 3> x_t;
    mutable etl::dyn_matrix<float, 3> s_t;

    void prepare_cache(size_t Batch, size_t time_steps, size_t sequence_length, size_t hidden_units) const {
        if (cpp_unlikely(!x_t.memory_start())) {
            x_t.resize(time_steps, Batch, sequence_length);
            s_t.resize(time_steps, Batch, hidden_units);
        }
    }

    /*!
     * \brief Apply the layer to the given batch of input.
     *
     * \param x A batch of input
     * \param output A batch of output that will be filled
     * \param w The W weights matrix
     * \param u The U weights matrix
     */
    template <typename H, typename V, typename W, typename U, typename B>
    void forward_batch_impl(H&& output, const V& x, const W& w, const U& u, const B& b, size_t time_steps, size_t sequence_length, size_t hidden_units) const {
        const auto Batch = etl::dim<0>(x);

        prepare_cache(Batch, time_steps, sequence_length, hidden_units);

        // 1. Rearrange input

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                x_t(t)(b) = x(b)(t);
            }
        }

        // 2. Forward propagation through time

        // t == 0

        s_t(0) = f_activate<activation_function>(bias_add_2d(x_t(0) * u, b));

        for (size_t t = 1; t < time_steps; ++t) {
            s_t(t) = f_activate<activation_function>(bias_add_2d(x_t(t) * u + s_t(t - 1) * w, b));
        }

        // 3. Rearrange the output

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                output(b)(t) = s_t(t)(b);
            }
        }
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template <typename H, typename C, typename W, typename U>
    void backward_batch_impl(H&& output, C& context, const W& w, const U& u, size_t time_steps, size_t sequence_length, size_t hidden_units, size_t bptt_steps,
                             bool direct = true) const {
        const size_t Batch = etl::dim<0>(context.errors);

        etl::dyn_matrix<float, 3> delta_t(time_steps, Batch, hidden_units);
        etl::dyn_matrix<float, 3> d_h_t(time_steps, Batch, hidden_units);
        etl::dyn_matrix<float, 3> d_x_t(time_steps, Batch, sequence_length);

        // 1. Rearrange errors

        for (size_t b = 0; b < Batch; ++b) {
            for (size_t t = 0; t < time_steps; ++t) {
                delta_t(t)(b) = context.errors(b)(t);
            }
        }

        // 2. Get the gradients from the context

        auto& w_grad = std::get<0>(context.up.context)->grad;
        auto& u_grad = std::get<1>(context.up.context)->grad;
        auto& b_grad = std::get<2>(context.up.context)->grad;

        w_grad = 0;
        u_grad = 0;
        b_grad = 0;

        // 3. Backpropagation through time

        size_t ttt = time_steps - 1;

        do {
            const size_t last_step = std::max(int(time_steps) - int(bptt_steps), 0);

            // Backpropagation through time
            for (int tt = ttt; tt >= int(last_step); --tt) {
                const size_t t = tt;

                if(t == time_steps - 1){
                    d_h_t(t) = delta_t(t) >> f_derivative<activation_function>(s_t(t));
                } else {
                    d_h_t(t) = (delta_t(t) + d_h_t(t + 1)) >> f_derivative<activation_function>(s_t(t));
                }

                if (t > 0) {
                    w_grad += etl::batch_outer(s_t(t - 1), d_h_t(t));
                }

                u_grad += etl::batch_outer(x_t(t), d_h_t(t));
                b_grad += etl::bias_batch_sum_2d(d_h_t(t));

                // Gradients to the input
                d_x_t(t) = d_h_t(t) * trans(u);

                // Update for next steps
                d_h_t(t) = d_h_t(t) * trans(w);
            }

            --ttt;

            // If only the last time step is used, no need to use the other errors
            if constexpr (desc::parameters::template contains<last_only>()) {
                break;
            }
        } while (ttt != 0);

        // 3. Rearrange for the output

        if (direct) {
            for (size_t b = 0; b < Batch; ++b) {
                for (size_t t = 0; t < time_steps; ++t) {
                    output(b)(t) = d_x_t(t)(b);
                }
            }
        }
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template <typename C, typename W, typename U>
    void compute_gradients_impl([[maybe_unused]] C& context, [[maybe_unused]] const W& w, [[maybe_unused]] const U& u, [[maybe_unused]] size_t time_steps,
                                [[maybe_unused]] size_t sequence_length, [[maybe_unused]] size_t hidden_units, [[maybe_unused]] size_t bptt_steps) const {
        if constexpr (!C::layer){
            backward_batch_impl(x_t, context, w, u, time_steps, sequence_length, hidden_units, bptt_steps, false);
        }
    }

    /*!
     * \brief Backup the weights in the secondary weights matrix
     */
    void backup_weights() {
        unique_safe_get(as_derived().bak_w) = as_derived().w;
        unique_safe_get(as_derived().bak_u) = as_derived().u;
        unique_safe_get(as_derived().bak_b) = as_derived().b;
    }

    /*!
     * \brief Restore the weights from the secondary weights matrix
     */
    void restore_weights() {
        as_derived().w = *as_derived().bak_w;
        as_derived().u = *as_derived().bak_u;
        as_derived().b = *as_derived().bak_b;
    }

    /*!
     * \brief Load the weigts into the given stream
     */
    void store(std::ostream& os) const {
        cpp::binary_write_all(os, as_derived().w);
        cpp::binary_write_all(os, as_derived().u);
        cpp::binary_write_all(os, as_derived().b);
    }

    /*!
     * \brief Load the weigts from the given stream
     */
    void load(std::istream& is) {
        cpp::binary_load_all(is, as_derived().w);
        cpp::binary_load_all(is, as_derived().u);
        cpp::binary_load_all(is, as_derived().b);
    }

    /*!
     * \brief Load the weigts into the given file
     */
    void store(const std::string& file) const {
        std::ofstream os(file, std::ofstream::binary);
        store(os);
    }

    /*!
     * \brief Load the weigts from the given file
     */
    void load(const std::string& file) {
        std::ifstream is(file, std::ifstream::binary);
        load(is);
    }

    /*!
     * \brief Returns the trainable variables of this layer.
     * \return a tuple containing references to the variables of this layer
     */
    decltype(auto) trainable_parameters() {
        return std::make_tuple(std::ref(as_derived().w), std::ref(as_derived().u), std::ref(as_derived().b));
    }

    /*!
     * \brief Returns the trainable variables of this layer.
     * \return a tuple containing references to the variables of this layer
     */
    decltype(auto) trainable_parameters() const {
        return std::make_tuple(std::cref(as_derived().w), std::cref(as_derived().u), std::cref(as_derived().b));
    }

private:
    //CRTP Deduction

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() {
        return *static_cast<derived_t*>(this);
    }

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const derived_t& as_derived() const {
        return *static_cast<const derived_t*>(this);
    }
};

} //end of dll namespace
