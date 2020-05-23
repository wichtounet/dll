//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cmath>
#include <vector>
#include <random>
#include <functional>
#include <ctime>

#include "cpp_utils/stop_watch.hpp" //Performance counter
#include "cpp_utils/assert.hpp"

#include "etl/etl.hpp"

#include "dll/util/checks.hpp"    //NaN checks
#include "dll/util/timers.hpp"    //auto_timer
#include "dll/rbm/rbm_base.hpp"       //The base class
#include "dll/base_conf.hpp"      //Descriptor configuration
#include "dll/rbm/rbm_tmp.hpp"        // static_if macros

namespace dll {

/*!
 * \brief Standard version of Restricted Boltzmann Machine
 *
 * This follows the definition of a RBM by Geoffrey Hinton. This is an "abstract" class,
 * using CRTP technique to inject functions into its children.
 */
template <typename Parent, typename Desc>
struct standard_rbm : public rbm_base<Parent, Desc> {
    using desc      = Desc;                         ///< The descriptor of the layer
    using parent_t  = Parent;                       ///< The type of the parent layer (CRTP)
    using this_type = standard_rbm<parent_t, desc>; ///< The type of this layer
    using base_type = rbm_base<parent_t, desc>;     ///< The base type
    using weight    = typename desc::weight;        ///< The data type for this layer

    using input_one_t  = typename rbm_base_traits<parent_t>::input_one_t;  ///< The type of one input
    using output_one_t = typename rbm_base_traits<parent_t>::output_one_t; ///< The type of one output
    using input_t      = typename rbm_base_traits<parent_t>::input_t;      ///< The type of the input
    using output_t     = typename rbm_base_traits<parent_t>::output_t;     ///< The type of the output

    static constexpr unit_type visible_unit = desc::visible_unit; ///< The type of visible unit
    static constexpr unit_type hidden_unit  = desc::hidden_unit;  ///< The type of hidden unit

    static_assert(visible_unit != unit_type::SOFTMAX, "Softmax Visible units are not support");
    static_assert(hidden_unit != unit_type::GAUSSIAN, "Gaussian hidden units are not supported");

    /*!
     * \brief Construct empty standard_rbm
     */
    standard_rbm() {
        //Better initialization of learning rate
        base_type::learning_rate =
            visible_unit == unit_type::GAUSSIAN && is_relu(hidden_unit) ? 1e-5
                                                                        : visible_unit == unit_type::GAUSSIAN || is_relu(hidden_unit) ? 1e-3
                                                                                                                                      : /* Only ReLU and Gaussian Units needs lower rate */ 1e-1;
    }

    //Energy functions

    /*!
     * \brief Return the energy of the given joint configuration
     * \param v The configuration of the inputs
     * \param h The configuration of the outputs
     * \return The scalar energy of the model for the given joint configuration
     */
    template<typename Input>
    weight energy(const Input& v, const output_one_t& h) const {
        return energy(as_derived(), v, h);
    }

    /*!
     * \brief Return the free energy of the given input
     */
    template<typename Input>
    weight free_energy(const Input& v) const {
        return free_energy(as_derived(), v);
    }

    /*!
     * \brief Return the free energy of the current input
     */
    weight free_energy() const {
        auto& rbm = as_derived();
        return free_energy(rbm, rbm.v1);
    }

    //Various functions

    /*!
     * \brief Initalize the weights using the training inputs
     */
    template <typename Iterator>
    void init_weights(Iterator&& first, Iterator&& last) {
        init_weights(std::forward<Iterator>(first), std::forward<Iterator>(last), as_derived());
    }

    /*!
     * \brief Initalize the weights using the training inputs
     */
    template <typename Generator>
    void init_weights(Generator& generator) {
        init_weights(generator, as_derived());
    }

    /*!
     * \brief Reconstruct the given input
     */
    void reconstruct(const input_one_t& items) {
        reconstruct(items, as_derived());
    }

    // activate_hidden

    using base_type::activate_hidden;

    /*!
     * \brief Compute the hidden representation from the given input.
     *
     * Special functions to be used by optimizer.
     *
     * \param h_a The output to set the activation probabilities of the hidden representation
     * \param h_s The output to set the activation samples of the hidden representation
     * \param v_a The input activation probabilities of the visible representation
     * \param v_s The input the activation samples of the visible representation
     * \param b The biases
     * \param w The weights
     */
    template <bool P = true, bool S = true, typename H1, typename H2, typename V, typename B, typename W>
    void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s, const B& b, const W& w) const {
        std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w);
    }

    /*!
     * \brief Compute the hidden representation from the given input
     * \param h_a The output to set the activation probabilities of the hidden representation
     * \param h_s The output to set the activation samples of the hidden representation
     * \param v_a The input activation probabilities of the visible representation
     * \param v_s The input the activation samples of the visible representation
     */
    template <bool P = true, bool S = true, typename H1, typename H2, typename V>
    void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s) const {
        std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, as_derived().b, as_derived().w);
    }

    /*!
     * \brief Compute the hidden representation from the given input
     * \param h_a The output to set
     * \param v_a The input
     */
    template <typename H, typename Input>
    void activate_hidden(H&& h_a, const Input& v_a) const {
        std_activate_hidden<true, false>(std::forward<H>(h_a), std::forward<H>(h_a), v_a, v_a, as_derived().b, as_derived().w);
    }

    // activate_visible

    template <bool P = true, bool S = true, typename H, typename V>
    void activate_visible(const H& h_a, const H& h_s, V&& v_a, V&& v_s) const {
        std_activate_visible<P, S>(h_a, h_s, std::forward<V>(v_a), std::forward<V>(v_s), as_derived().c, as_derived().w);
    }

    // batch_activate_visible

    template <bool P = true, bool S = true, typename H, typename V>
    void batch_activate_visible(const H& h_a, const H& h_s, V&& v_a, V&& v_s) const {
        batch_std_activate_visible<P, S>(h_a, h_s, std::forward<V>(v_a), std::forward<V>(v_s), as_derived().c, as_derived().w);
    }

    // batch_activate_hidden

    /*!
     * \brief Compute the hidden representation from the given input
     * \param h_a The batch output to set the activation probabilities of the hidden representation
     * \param h_s The batch output to set the activation samples of the hidden representation
     * \param v_a The batch input activation probabilities of the visible representation
     * \param v_s The batch input the activation samples of the visible representation
     */
    template <bool P = true, bool S = true, typename H1, typename H2, typename V>
    void batch_activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s) const {
        batch_std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, as_derived().b, as_derived().w);
    }

    /*!
     * \brief Compute the hidden representation from a given batch of input
     *
     * \param h_a The batch output to set
     * \param v_a The batch input
     */
    template <typename H, typename V>
    void batch_activate_hidden(H&& h_a, const V& v_a) const {
        if constexpr (etl::decay_traits<V>::dimensions() == 2) {
            batch_std_activate_hidden<true, false>(std::forward<H>(h_a), std::forward<H>(h_a), v_a, v_a, as_derived().b, as_derived().w);
        } else {
            batch_std_activate_hidden<true, false>(std::forward<H>(h_a), std::forward<H>(h_a),
                                                   etl::reshape(v_a, etl::dim(h_a, 0), as_derived().input_size()),
                                                   etl::reshape(v_a, etl::dim(h_a, 0), as_derived().input_size()), as_derived().b, as_derived().w);
        }
    }

    //Display functions

    /*!
     * \brief Display the values of the units of the RBM
     */
    void display_units() const {
        display_visible_units();
        display_hidden_units();
    }

    /*!
     * \brief Display the values of the visible units of the RBM
     */
    void display_visible_units() const {
        display_visible_units(as_derived());
    }

    /*!
     * \brief Display the values of the visible units of the RBM using the given
     * size to display a square matrix of the values.
     */
    void display_visible_units(size_t matrix) const {
        display_visible_units(as_derived(), matrix);
    }

    /*!
     * \brief Display the values of the hidden units of the RBM
     */
    void display_hidden_units() const {
        display_hidden_units(as_derived());
    }

    /*!
     * \brief Display the weights of the RBM.
     */
    void display_weights() const {
        display_weights(as_derived());
    }

    /*!
     * \brief Display the weights of the RBM using the given
     * size to display a square matrix of the values.
     */
    void display_weights(size_t matrix) const {
        display_weights(matrix, as_derived());
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    output_t prepare_output(size_t samples, bool is_last = false, size_t labels = 0) const {
        output_t output;
        output.reserve(samples);

        for (size_t i = 0; i < samples; ++i) {
            output.emplace_back(as_derived().output_size() + (is_last ? labels : 0));
        }

        return output;
    }

    template <typename Input>
    output_one_t prepare_one_output(bool is_last = false, size_t labels = 0) const {
        return output_one_t(as_derived().output_size() + (is_last ? labels : 0));
    }

    input_one_t prepare_one_input() const {
        return input_one_t(as_derived().input_size());
    }

    friend base_type;

private:
    //Since the sub classes does not have the same fields, it is not possible
    //to put the fields in standard_rbm, therefore, it is necessary to use template
    //functions to implement the details

    template<typename V>
    static double reconstruction_error_impl(const V& items, parent_t& rbm) {
        cpp_assert(items.size() == num_visible(rbm), "The size of the training sample must match visible units");

        //Set the state of the visible units
        rbm.v1 = items;

        rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);
        rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);

        return etl::mean((rbm.v1 - rbm.v2_a) >> (rbm.v1 - rbm.v2_a));
    }

    /*!
     * \brief Initalize the weights using the training inputs
     */
    template <typename Generator>
    static void init_weights(Generator& generator, parent_t& rbm) {
        const auto size = generator.size();

        //Initialize the visible biases to log(pi/(1-pi))
        for (size_t i = 0; i < num_visible(rbm); ++i) {
            generator.reset();

            size_t count = 0;

            while(generator.has_next_batch()){
                auto labels = generator.label_batch();

                for(size_t b = 0; b < etl::dim<0>(labels); ++b){
                    if(labels(b)[i] == 1){
                        ++count;
                    }
                }

                generator.next_batch();
            }

            auto pi = static_cast<double>(count) / size;
            pi += 0.0001;
            rbm.c(i) = log(pi / (1.0 - pi));

            cpp_assert(std::isfinite(rbm.c(i)), "NaN verify");
        }
    }

    /*!
     * \brief Initalize the weights using the training inputs
     */
    template <typename Iterator>
    static void init_weights(Iterator first, Iterator last, parent_t& rbm) {
        auto size = std::distance(first, last);

        //Initialize the visible biases to log(pi/(1-pi))
        for (size_t i = 0; i < num_visible(rbm); ++i) {
            auto count = std::count_if(first, last,
                                       [i](auto& a) { return a[i] == 1; });

            auto pi = static_cast<double>(count) / size;
            pi += 0.0001;
            rbm.c(i) = log(pi / (1.0 - pi));

            cpp_assert(std::isfinite(rbm.c(i)), "NaN verify");
        }
    }

    /*!
     * \brief Reconstruct the given input
     */
    static void reconstruct(const input_one_t& items, parent_t& rbm) {
        cpp_assert(items.size() == num_visible(rbm), "The size of the training sample must match visible units");

        cpp::stop_watch<> watch;

        //Set the state of the visible units
        rbm.v1 = items;

        rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);
        rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);
        rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);

        std::cout << "Reconstruction took " << watch.elapsed() << "ms" << std::endl;
    }

    /*!
     * \brief Display the weights of the RBM
     */
    static void display_weights(const parent_t& rbm) {
        for (size_t j = 0; j < num_hidden(rbm); ++j) {
            for (size_t i = 0; i < num_visible(rbm); ++i) {
                std::cout << rbm.w(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    /*!
     * \brief Display the weights of the RBM
     */
    static void display_weights(const parent_t& rbm, size_t matrix) {
        for (size_t j = 0; j < num_hidden(rbm); ++j) {
            for (size_t i = 0; i < num_visible(rbm);) {
                for (size_t m = 0; m < matrix; ++m) {
                    std::cout << rbm.w(i++, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    /*!
     * \brief Display the current visible unit samples
     */
    static void display_visible_units(const parent_t& rbm) {
        std::cout << "Visible  Value" << std::endl;

        for (size_t i = 0; i < num_visible(rbm); ++i) {
            printf("%-8lu %d\n", i, rbm.v2_s(i));
        }
    }

    /*!
     * \brief Display the current visible unit samples
     */
    static void display_visible_units(const parent_t& rbm, size_t matrix) {
        for (size_t i = 0; i < matrix; ++i) {
            for (size_t j = 0; j < matrix; ++j) {
                std::cout << rbm.v2_s(i * matrix + j) << " ";
            }
            std::cout << std::endl;
        }
    }

    /*!
     * \brief Display the current visible unit samples
     */
    static void display_hidden_units(const parent_t& rbm) {
        std::cout << "Hidden Value" << std::endl;

        for (size_t j = 0; j < num_hidden(rbm); ++j) {
            printf("%-8lu %d\n", j, rbm.h2_s(j));
        }
    }

    //Note: Considering that energy and free energy are not critical, their implementations
    //are not highly optimized.

    template <typename V, typename H>
    weight energy(const parent_t& rbm, const V& v, const H& h) const {
        auto rv = reshape(v, as_derived().num_visible);

        if constexpr (visible_unit == unit_type::BINARY && hidden_unit == unit_type::BINARY) {
            //Definition according to G. Hinton
            //E(v,h) = -sum(ai*vi) - sum(bj*hj) -sum(vi*hj*wij)

            auto x = rbm.b + rv * rbm.w;

            return -etl::dot(rbm.c, rv) - etl::dot(rbm.b, h) - etl::sum(x);
        } else if constexpr (visible_unit == unit_type::GAUSSIAN && hidden_unit == unit_type::BINARY) {
            //Definition according to G. Hinton
            //E(v,h) = -sum((vi - ai)^2/(2*var*var)) - sum(bj*hj) -sum((vi/var)*hj*wij)

            auto x = rbm.b + rv * rbm.w;

            return etl::sum(etl::pow(rv - rbm.c, 2) / 2.0) - etl::dot(rbm.b, h) - etl::sum(x);
        } else {
            return 0.0;
        }
    }

    //Free energy are computed from the E(v,h) formulas
    //1. by isolating hi in the E(v,h) formulas
    //2. by using the sum_hi which sums over all the possible values of hi
    //3. by considering only binary hidden units, the values are only 0 and 1
    //and therefore the values can be "integrated out" easily.

    template <typename V>
    weight free_energy(const parent_t& rbm, const V& v) const {
        auto rv = reshape(v, as_derived().num_visible);

        if constexpr (visible_unit == unit_type::BINARY && hidden_unit == unit_type::BINARY) {
            //Definition according to G. Hinton
            //F(v) = -sum(ai*vi) - sum(log(1 + e^(xj)))

            auto x = rbm.b + rv * rbm.w;

            return -etl::dot(rbm.c, rv) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else if constexpr (visible_unit == unit_type::GAUSSIAN && hidden_unit == unit_type::BINARY) {
            //Definition computed from E(v,h)
            //F(v) = sum((vi-ai)^2/2) - sum(log(1 + e^(xj)))

            auto x = rbm.b + rv * rbm.w;

            return etl::sum(etl::pow(rv - rbm.c, 2) / 2.0) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else {
            return 0.0;
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V, typename B, typename W>
    void std_activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V&, const B& b, const W& w) const {
        if constexpr (etl::decay_traits<V>::dimensions() == 1) {
            dll::auto_timer timer("rbm:std:activate_hidden");

            using namespace etl;

            //Compute activation probabilities

            if constexpr (P && hidden_unit == unit_type::BINARY) {
                h_a = etl::sigmoid(b + (v_a * w));
            }

            if constexpr (P && hidden_unit == unit_type::RELU) {
                h_a = max(b + (v_a * w), 0.0);
            }

            if constexpr (P && hidden_unit == unit_type::RELU1) {
                h_a = min(max(b + (v_a * w), 0.0), 1.0);
            }

            if constexpr (P && hidden_unit == unit_type::RELU6) {
                h_a = min(max(b + (v_a * w), 0.0), 6.0);
            }

            if constexpr (P && hidden_unit == unit_type::SOFTMAX) {
                h_a = stable_softmax(b + (v_a * w));
            }

            // Sample the values from the probs

            if constexpr (P && S && hidden_unit == unit_type::BINARY) {
                h_s = bernoulli(h_a);
            }

            if constexpr (P && S && hidden_unit == unit_type::RELU) {
                h_s = max(logistic_noise(b + (v_a * w)), 0.0);
            }

            if constexpr (P && S && hidden_unit == unit_type::RELU1) {
                h_s = min(max(ranged_noise(b + (v_a * w), 1.0), 0.0), 1.0);
            }

            if constexpr (P && S && hidden_unit == unit_type::RELU6) {
                h_s = min(max(ranged_noise(b + (v_a * w), 6.0), 0.0), 6.0);
            }

            if constexpr (P && S && hidden_unit == unit_type::SOFTMAX) {
                h_s = one_if_max(h_a);
            }

            // Sample the values directly from the input

            if constexpr (!P && S && hidden_unit == unit_type::BINARY) {
                h_s = bernoulli(etl::sigmoid(b + (v_a * w)));
            }

            if constexpr (!P && S && hidden_unit == unit_type::RELU) {
                h_s = max(logistic_noise(b + (v_a * w)), 0.0);
            }

            if constexpr (!P && S && hidden_unit == unit_type::RELU1) {
                h_s = min(max(ranged_noise(b + (v_a * w), 1.0), 0.0), 1.0);
            }

            if constexpr (!P && S && hidden_unit == unit_type::RELU6) {
                min(max(ranged_noise(b + (v_a * w), 6.0), 0.0), 6.0);
            }

            if constexpr (!P && S && hidden_unit == unit_type::SOFTMAX) {
                h_s = one_if_max(stable_softmax(b + (v_a * w)));
            }

            // NaN Checks

            if (P) {
                nan_check_deep(h_a);
            }

            if (S) {
                nan_check_deep(h_s);
            }
        } else {
            auto r = reshape(v_a, as_derived().num_visible);
            std_activate_hidden(h_a, h_s, r, r, b, w);
        }
    }

    template <bool P = true, bool S = true, typename H, typename V, typename C, typename W>
    static void std_activate_visible(const H&, const H& h_s, V&& v_a, V&& v_s, const C& c, const W& w) {
        dll::auto_timer timer("rbm:std:activate_visible");

        using namespace etl;

        // Compute the visible activation probabilities

        if constexpr (P && visible_unit == unit_type::BINARY) {
            v_a = etl::sigmoid(c + (w * h_s));
        }

        if constexpr (P && visible_unit == unit_type::GAUSSIAN) {
            v_a = c + (w * h_s);
        }

        if constexpr (P && visible_unit == unit_type::RELU) {
            v_a = max(c + (w * h_s), 0.0);
        }

        // Sample the visible values from the input

        if constexpr (!P && S && visible_unit == unit_type::BINARY) {
            v_s = bernoulli(etl::sigmoid(c + (w * h_s)));
        }

        if constexpr (!P && S && visible_unit == unit_type::GAUSSIAN) {
            v_s = normal_noise(c + (w * h_s));
        }

        if constexpr (!P && S && visible_unit == unit_type::RELU) {
            v_s = logistic_noise(max(c + (w * h_s), 0.0));
        }

        // NaN Checks

        if (P) {
            nan_check_deep(v_a);
        }

        if (S) {
            nan_check_deep(v_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V, typename B, typename W>
    static void batch_std_activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V&, const B& b, const W& w) {
        dll::auto_timer timer("rbm:std:batch_activate_hidden");

        using namespace etl;

        const auto Batch = etl::dim<0>(h_a);

        cpp_assert(etl::dim<0>(h_s) == Batch && etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");

        // Compute activation probabilities

        if constexpr (P && hidden_unit == unit_type::BINARY) {
            h_a = etl::sigmoid(rep_l(b, Batch) + v_a * w);
        }

        if constexpr (P && hidden_unit == unit_type::RELU) {
            h_a = max(rep_l(b, Batch) + v_a * w, 0.0);
        }

        if constexpr (P && hidden_unit == unit_type::RELU1) {
            h_a = min(max(rep_l(b, Batch) + v_a * w, 0.0), 1.0);
        }

        if constexpr (P && hidden_unit == unit_type::RELU6) {
            h_a = min(max(rep_l(b, Batch) + v_a * w, 0.0), 6.0);
        }

        if constexpr (P && hidden_unit == unit_type::SOFTMAX) {
            auto x = etl::force_temporary(rep_l(b, Batch) + v_a * w);

            for (size_t b = 0; b < Batch; ++b) {
                h_a(b) = stable_softmax(x(b));
            }
        }

        // Samples values from the probabilities

        if constexpr (P && S && hidden_unit == unit_type::BINARY) {
            h_s = bernoulli(h_a);
        }

        if constexpr (P && S && hidden_unit == unit_type::RELU) {
            h_s = max(logistic_noise(rep_l(b, Batch) + v_a * w), 0.0);
        }

        if constexpr (P && S && hidden_unit == unit_type::RELU1) {
            h_s = min(max(ranged_noise(rep_l(b, Batch) + v_a * w, 1.0), 0.0), 1.0);
        }

        if constexpr (P && S && hidden_unit == unit_type::RELU6) {
            h_s = min(max(ranged_noise(rep_l(b, Batch) + v_a * w, 6.0), 0.0), 6.0);
        }

        if constexpr (P && S && hidden_unit == unit_type::SOFTMAX) {
            for (size_t b = 0; b < Batch; ++b) {
                h_s(b) = one_if_max(h_a(b));
            }
        }

        // Sample values directly from the input

        if constexpr (!P && S && hidden_unit == unit_type::BINARY) {
            h_s = bernoulli(etl::sigmoid(rep_l(b, Batch) + v_a * w));
        }

        if constexpr (!P && S && hidden_unit == unit_type::RELU) {
            h_s = max(logistic_noise(rep_l(b, Batch) + v_a * w), 0.0);
        }

        if constexpr (!P && S && hidden_unit == unit_type::RELU1) {
            h_s = min(max(ranged_noise(rep_l(b, Batch) + v_a * w, 1.0), 0.0), 1.0);
        }

        if constexpr (!P && S && hidden_unit == unit_type::RELU6) {
            h_s = min(max(ranged_noise(rep_l(b, Batch) + v_a * w, 6.0), 0.0), 6.0);
        }

        if constexpr (!P && S && hidden_unit == unit_type::SOFTMAX) {
            auto x = etl::force_temporary(rep_l(b, Batch) + v_a * w);

            for (size_t b = 0; b < Batch; ++b) {
                h_s(b) = one_if_max(stable_softmax(x(b)));
            }
        }

        // NaN Checks

        if (P) {
            nan_check_deep(h_a);
        }

        if (S) {
            nan_check_deep(h_s);
        }
    }

    template <bool P = true, bool S = true, typename H, typename V, typename C, typename W>
    static void batch_std_activate_visible(const H&, const H& h_s, V&& v_a, V&& v_s, const C& c, const W& w) {
        dll::auto_timer timer("rbm:std:batch_activate_visible");

        using namespace etl;

        const auto Batch = etl::dim<0>(v_s);

        cpp_assert(etl::dim<0>(h_s) == Batch && etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");

        // Compute the visible activation probabilities

        if constexpr (P && visible_unit == unit_type::BINARY) {
            v_a = etl::sigmoid(rep_l(c, Batch) + transpose(w * transpose(h_s)));
        }

        if constexpr (P && visible_unit == unit_type::GAUSSIAN) {
            v_a = rep_l(c, Batch) + transpose(w * transpose(h_s));
        }

        if constexpr (P && visible_unit == unit_type::RELU) {
            v_a = max(rep_l(c, Batch) + transpose(w * transpose(h_s)), 0.0);
        }

        // Sample the visible values from the input

        if constexpr (!P && S && visible_unit == unit_type::BINARY) {
            v_s = bernoulli(etl::sigmoid(rep_l(c, Batch) + transpose(w * transpose(h_s))));
        }

        if constexpr (!P && S && visible_unit == unit_type::GAUSSIAN) {
            v_s = normal_noise(rep_l(c, Batch) + transpose(w * transpose(h_s)));
        }

        if constexpr (!P && S && visible_unit == unit_type::RELU) {
            v_s = logistic_noise(max(rep_l(c, Batch) + transpose(w * transpose(h_s)), 0.0));
        }

        // NaN Checks

        if (P) {
            nan_check_deep(v_a);
        }

        if (S) {
            nan_check_deep(v_s);
        }
    }

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    parent_t& as_derived() {
        return *static_cast<parent_t*>(this);
    }

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const parent_t& as_derived() const {
        return *static_cast<const parent_t*>(this);
    }
};

} //end of dll namespace
