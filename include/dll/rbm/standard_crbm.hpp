//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementation of a Convolutional Restricted Boltzmann Machine
 */

#pragma once

#include <cstddef>
#include <ctime>
#include <random>

#include "cpp_utils/assert.hpp"     //Assertions
#include "cpp_utils/stop_watch.hpp" //Performance counter
#include "cpp_utils/maybe_parallel.hpp"

#include "etl/etl.hpp"

#include "standard_conv_rbm.hpp" //The base class
#include "rbm_tmp.hpp"           // static_if macros

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template <typename Derived, typename Desc>
struct standard_crbm : public standard_conv_rbm<Derived, Desc> {
    using derived_t = Derived;                          ///< The derived type
    using desc      = Desc;                             ///< The descriptor of the layer
    using weight    = typename desc::weight;            ///< The data type for this layer
    using this_type = standard_crbm<derived_t, desc>;   ///< The type of this layer
    using base_type = standard_conv_rbm<Derived, desc>; ///< The base type
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    using input_one_t  = typename rbm_base_traits<derived_t>::input_one_t;  ///< The type of one input
    using output_one_t = typename rbm_base_traits<derived_t>::output_one_t; ///< The type of one output
    using input_t      = typename rbm_base_traits<derived_t>::input_t;      ///< The type of the input
    using output_t     = typename rbm_base_traits<derived_t>::output_t;     ///< The type of the output

    static constexpr unit_type visible_unit = desc::visible_unit; ///< The visible unit type
    static constexpr unit_type hidden_unit  = desc::hidden_unit;  ///< The hidden unit type

    standard_crbm() = default;

    // Make base class them participate in overload resolution
    using base_type::activate_hidden;
    using base_type::batch_activate_hidden;

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& /*v_s*/) const {
        dll::auto_timer timer("crbm:activate_hidden");

        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        as_derived().template validate_outputs<H1, H2>();

        using namespace etl;

        // TODO This code is a huge mess!

        auto b_rep = as_derived().get_b_rep();

        as_derived().reshape_h_a(h_a) = etl::conv_4d_valid_flipped(as_derived().reshape_v_a(v_a), as_derived().w);

        // Need to be done before h_a is computed!

        if constexpr (P && S && hidden_unit == unit_type::RELU) {
            h_s = max(logistic_noise(b_rep + h_a), 0.0);
        }

        if constexpr (P && S && hidden_unit == unit_type::RELU1) {
            h_s = min(max(ranged_noise(b_rep + h_a, 1.0), 0.0), 1.0);
        }

        if constexpr (P && S && hidden_unit == unit_type::RELU6) {
            h_s = min(max(ranged_noise(b_rep + h_a, 6.0), 0.0), 6.0);
        }

        if constexpr (P && hidden_unit == unit_type::BINARY && visible_unit == unit_type::BINARY) {
            h_a = etl::sigmoid(b_rep + h_a);
        }

        if constexpr (P && hidden_unit == unit_type::BINARY && visible_unit == unit_type::GAUSSIAN) {
            h_a = etl::sigmoid((1.0 / (0.1 * 0.1)) >> (b_rep + h_a));
        }

        if constexpr (P && hidden_unit == unit_type::RELU) {
            h_a = max(b_rep + h_a, 0.0);
        }

        if constexpr (P && hidden_unit == unit_type::RELU1) {
            h_a = min(max(b_rep + h_a, 0.0), 1.0);
        }

        if constexpr (P && hidden_unit == unit_type::RELU6) {
            h_a = min(max(b_rep + h_a, 0.0), 6.0);
        }

        if constexpr (P && S && hidden_unit == unit_type::BINARY) {
            h_s = bernoulli(h_a);
        }

        // NaN checks

        if (P) {
            nan_check_deep(h_a);
        }

        if (S) {
            nan_check_deep(h_s);
        }
    }

    template<typename Input>
    void activate_hidden(output_one_t& h_a, const Input& input) const {
        activate_hidden<true, false>(h_a, h_a, input, input);
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_visible(const H1& /*h_a*/, const H2& h_s, V1&& v_a, V2&& v_s) const {
        dll::auto_timer timer("crbm:activate_visible");

        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        as_derived().template validate_outputs<H1, H2>();

        using namespace etl;

        // Note: we reuse v_a as a temporary here, before adding the biases
        as_derived().reshape_v_a(v_a) = etl::conv_4d_full(as_derived().reshape_h_a(h_s), as_derived().w);

        auto c_rep = as_derived().get_c_rep();

        // Compute the activation probabilities

        if constexpr (P && visible_unit == unit_type::BINARY) {
            v_a = etl::sigmoid(c_rep + v_a);
        }

        if constexpr (P && visible_unit == unit_type::GAUSSIAN) {
            v_a = c_rep + v_a;
        }

        // Sample the values from the probabilities

        if constexpr (P && S && visible_unit == unit_type::BINARY) {
            v_s = bernoulli(v_a);
        }

        if constexpr (P && S && visible_unit == unit_type::GAUSSIAN) {
            v_s = normal_noise(v_a);
        }

        // NaN checks

        if (P) {
            nan_check_deep(v_a);
        }

        if (S) {
            nan_check_deep(v_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void batch_activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& /*v_s*/) const {
        dll::auto_timer timer("crbm:batch_activate_hidden");

        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        as_derived().template validate_outputs<H1, H2, 1>();

        using namespace etl;

        // TODO This code is a huge mess!

        h_a = etl::conv_4d_valid_flipped(v_a, as_derived().w);

        // Need to be done before h_a is computed!
        if constexpr (P && S && hidden_unit == unit_type::RELU) {
            h_s = max(logistic_noise(bias_add_4d(h_a, as_derived().b)), 0.0);
        }

        if constexpr (P && S && hidden_unit == unit_type::RELU1) {
            h_s = min(max(ranged_noise(bias_add_4d(h_a, as_derived().b), 1.0), 0.0), 1.0);
        }

        if constexpr (P && S && hidden_unit == unit_type::RELU6) {
            h_s = min(max(ranged_noise(bias_add_4d(h_a, as_derived().b), 6.0), 0.0), 6.0);
        }

        if constexpr (P && hidden_unit == unit_type::BINARY && visible_unit == unit_type::BINARY) {
            h_a = etl::sigmoid(bias_add_4d(h_a, as_derived().b));
        }

        if constexpr (P && hidden_unit == unit_type::BINARY && visible_unit == unit_type::GAUSSIAN) {
            h_a = etl::sigmoid((1.0 / (0.1 * 0.1)) >> (bias_add_4d(h_a, as_derived().b)));
        }

        if constexpr (P && hidden_unit == unit_type::RELU) {
            h_a = max(bias_add_4d(h_a, as_derived().b), 0.0);
        }

        if constexpr (P && hidden_unit == unit_type::RELU1) {
            h_a = min(max(bias_add_4d(h_a, as_derived().b), 0.0), 1.0);
        }

        if constexpr (P && hidden_unit == unit_type::RELU6) {
            h_a = min(max(bias_add_4d(h_a, as_derived().b), 0.0), 6.0);
        }

        if constexpr (P && S && hidden_unit == unit_type::BINARY) {
            h_s = bernoulli(h_a);
        }

        // NaN checks

        if (P) {
            nan_check_deep(h_a);
        }

        if (S) {
            nan_check_deep(h_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void batch_activate_visible(const H1& /*h_a*/, const H2& h_s, V1&& v_a, V2&& v_s) const {
        dll::auto_timer timer("crbm:batch_activate_visible");

        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        as_derived().template validate_outputs<H1, H2, 1>();

        // Note: we reuse v_a as a temporary here, before adding the biases
        v_a = etl::conv_4d_full(h_s, as_derived().w);

        // Compute the activation probabilities

        if constexpr (P && visible_unit == unit_type::BINARY) {
            v_a = etl::sigmoid(bias_add_4d(v_a, as_derived().c));
        }

        if constexpr (P && visible_unit == unit_type::GAUSSIAN) {
            v_a = bias_add_4d(v_a, as_derived().c);
        }

        // Sample the values from the probabilities

        if constexpr (P && S && visible_unit == unit_type::BINARY) {
            v_s = bernoulli(v_a);
        }

        if constexpr (P && S && visible_unit == unit_type::GAUSSIAN) {
            v_s = normal_noise(v_a);
        }

        // NaN Checks

        if (P) {
            nan_check_deep(v_a);
        }

        if (S) {
            nan_check_deep(v_s);
        }
    }

    friend base_type;

private:
    template<typename Input, typename Out>
    weight energy_impl(const Input& v, const Out& h) const {
        static_assert(etl::is_etl_expr<Out>, "energy_impl works with ETL expressions only");

        auto rv = as_derived().reshape_v_a(v);
        auto tmp = as_derived().energy_tmp();
        tmp = etl::conv_4d_valid_flipped(rv, as_derived().w);

        if constexpr (desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY) {
            //Definition according to Honglak Lee
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - c sum_v v

            return -etl::sum(as_derived().c >> etl::sum_r(rv(0))) - etl::sum(as_derived().b >> etl::sum_r(h)) - etl::sum(h >> tmp(0));
        } else if constexpr (desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY) {
            //Definition according to Honglak Lee / Mixed with Gaussian
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - sum_v ((v - c) ^ 2 / 2)

            auto c_rep = as_derived().get_c_rep();
            return -sum(etl::pow(rv(0) - c_rep, 2) / 2.0) - etl::sum(as_derived().b >> etl::sum_r(h)) - etl::sum(h >> tmp(0));
        } else {
            return 0.0;
        }
    }

    template<typename Input>
    weight free_energy_impl(const Input& v) const {
        auto rv = as_derived().reshape_v_a(v);
        auto tmp = as_derived().energy_tmp();
        tmp = etl::conv_4d_valid_flipped(rv, as_derived().w);

        if constexpr (desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY) {
            //Definition computed from E(v,h)

            auto b_rep = as_derived().get_b_rep();
            auto x = b_rep + tmp(0);
            return -etl::sum(as_derived().c >> etl::sum_r(rv(0))) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else if constexpr (desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY) {
            //Definition computed from E(v,h)

            auto b_rep = as_derived().get_b_rep();
            auto x = b_rep + tmp(0);
            auto c_rep = as_derived().get_c_rep();
            return -sum(etl::pow(rv(0) - c_rep, 2) / 2.0) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else {
            return 0.0;
        }
    }

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
