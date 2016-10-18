//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementation of a Convolutional Restricted Boltzmann Machine with Probabilistic Max Pooling
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
#include "base_conf.hpp"         //The configuration helpers
#include "util/timers.hpp"       //auto_timer
#include "util/checks.hpp"       //nan_check
#include "rbm_tmp.hpp"           // static_if macros

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine with Probabilistic
 * Max-Pooling.
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template <typename Derived, typename Desc>
struct standard_crbm_mp : public standard_conv_rbm<Derived, Desc> {
    using derived_t = Derived;
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = standard_crbm_mp<derived_t, desc>;
    using base_type = standard_conv_rbm<Derived, desc>;

    using input_one_t         = typename rbm_base_traits<derived_t>::input_one_t;
    using output_one_t        = typename rbm_base_traits<derived_t>::output_one_t;
    using input_t             = typename rbm_base_traits<derived_t>::input_t;
    using output_t            = typename rbm_base_traits<derived_t>::output_t;
    using hidden_output_one_t = typename rbm_base_traits<derived_t>::hidden_output_one_t;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit  = desc::hidden_unit;
    static constexpr const unit_type pooling_unit = desc::pooling_unit;

    standard_crbm_mp() : base_type() {
        // Nothing to init
    }

    // Make base class them participate in overload resolution
    using base_type::activate_hidden;

    size_t C() const {
        return as_derived().pool_C();
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2&) const {
        dll::auto_timer timer("crbm:mp:activate_hidden");

        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        auto b_rep = as_derived().get_b_rep();

        as_derived().reshape_h_a(h_a) = etl::conv_4d_valid_flipped(as_derived().reshape_v_a(v_a), as_derived().w);

        H_PROBS2(unit_type::BINARY, unit_type::BINARY, f(h_a) = etl::p_max_pool_h(b_rep + h_a, this->C(), this->C()));
        H_PROBS2(unit_type::BINARY, unit_type::GAUSSIAN, f(h_a) = etl::p_max_pool_h((1.0 / (0.1 * 0.1)) >> (b_rep + h_a), this->C(), this->C()));
        H_PROBS(unit_type::RELU, f(h_a) = f(h_a) = max(b_rep + h_a, 0.0));
        H_PROBS(unit_type::RELU6, f(h_a) = f(h_a) = min(max(b_rep + h_a, 0.0), 6.0));
        H_PROBS(unit_type::RELU1, f(h_a) = f(h_a) = min(max(b_rep + h_a, 0.0), 1.0));

        H_SAMPLE_PROBS(unit_type::BINARY, f(h_s) = bernoulli(h_a));
        H_SAMPLE_PROBS(unit_type::RELU, f(h_s) = max(logistic_noise(b_rep + h_a), 0.0));
        H_SAMPLE_PROBS(unit_type::RELU6, f(h_s) = ranged_noise(h_a, 6.0));
        H_SAMPLE_PROBS(unit_type::RELU1, f(h_s) = ranged_noise(h_a, 1.0));

        nan_check_etl(h_a);

        if (S) {
            nan_check_deep(h_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_visible(const H1&, const H2& h_s, V1&& v_a, V2&& v_s) const {
        dll::auto_timer timer("crbm:mp:activate_visible");

        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        using namespace etl;

        as_derived().reshape_v_a(v_a) = etl::conv_4d_full(as_derived().reshape_h_a(h_s), as_derived().w);

        auto c_rep = as_derived().get_c_rep();

        V_PROBS(unit_type::BINARY, f(v_a) = sigmoid(c_rep + v_a));
        V_PROBS(unit_type::GAUSSIAN, f(v_a) = c_rep + v_a);

        nan_check_deep(v_a);

        V_SAMPLE_PROBS(unit_type::BINARY, f(v_s) = bernoulli(v_a));
        V_SAMPLE_PROBS(unit_type::GAUSSIAN, f(v_s) = normal_noise(v_a));

        if (S) {
            nan_check_deep(v_s);
        }
    }

    template <bool P = true, bool S = true, typename Po, typename V>
    void activate_pooling(Po& p_a, Po& p_s, const V& v_a, const V&) const {
        dll::auto_timer timer("crbm:mp:activate_pooling");

        static_assert(pooling_unit == unit_type::BINARY, "Invalid pooling unit type");
        static_assert(P, "Computing S without P is not implemented");

        auto b_rep = as_derived().get_b_rep();

        auto v_cv = as_derived().energy_tmp();
        v_cv = etl::conv_4d_valid_flipped(as_derived().reshape_v_a(v_a), as_derived().w);

        if (pooling_unit == unit_type::BINARY) {
            p_a = etl::p_max_pool_p(b_rep + v_cv(0), C(), C());
        }

        nan_check_etl(p_a);

        if (S) {
            if (pooling_unit == unit_type::BINARY) {
                p_s = r_bernoulli(p_a);
            }

            nan_check_etl(p_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void batch_activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2&) const {
        dll::auto_timer timer("crbm:mp:batch_activate_hidden");

        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        const auto Batch = etl::dim<0>(h_a);

        cpp_assert(etl::dim<0>(h_s) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");
        cpp_unused(Batch);

        h_a = etl::conv_4d_valid_flipped(v_a, as_derived().w);

        auto b_rep = as_derived().get_batch_b_rep(v_a);

        // The loop is necessary because p_max_pool only handles 2/3D
        for (size_t i = 0; i < Batch; ++i) {
            H_PROBS2(unit_type::BINARY, unit_type::BINARY, f(h_a)(i) = etl::p_max_pool_h(b_rep + h_a(i), this->C(), this->C()));
            H_PROBS2(unit_type::BINARY, unit_type::GAUSSIAN, f(h_a)(i) = etl::p_max_pool_h((1.0 / (0.1 * 0.1)) >> (b_rep + h_a(i)), this->C(), this->C()));
            H_PROBS(unit_type::RELU, f(h_a)(i) = max(b_rep + h_a(i), 0.0));
            H_PROBS(unit_type::RELU6, f(h_a)(i) = min(max(b_rep + h_a(i), 0.0), 6.0));
            H_PROBS(unit_type::RELU1, f(h_a)(i) = min(max(b_rep + h_a(i), 0.0), 1.0));

            H_SAMPLE_PROBS(unit_type::RELU, f(h_s)(i) = max(logistic_noise(b_rep + h_a(i)), 0.0));
        }

        H_SAMPLE_PROBS(unit_type::BINARY, f(h_s) = bernoulli(h_a));
        H_SAMPLE_PROBS(unit_type::RELU6, f(h_s) = ranged_noise(h_a, 6.0));
        H_SAMPLE_PROBS(unit_type::RELU1, f(h_s) = ranged_noise(h_a, 1.0));

        nan_check_deep(h_a);

        if (S) {
            nan_check_deep(h_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void batch_activate_visible(const H1& h_a, const H2& h_s, V1&& v_a, V2&& v_s) const {
        dll::auto_timer timer("crbm:mp:batch_activate_visible");

        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        v_a = etl::conv_4d_full(h_s, as_derived().w);

        auto c_rep = as_derived().get_batch_c_rep(h_s);

        const auto Batch = etl::dim<0>(h_a);
        cpp_assert(etl::dim<0>(h_s) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_s) == Batch, "The number of batch must be consistent");
        cpp_unused(Batch);

        V_PROBS(unit_type::BINARY, f(v_a) = etl::sigmoid(c_rep + v_a));
        V_PROBS(unit_type::GAUSSIAN, f(v_a) = c_rep + v_a);

        V_SAMPLE_PROBS(unit_type::BINARY, f(v_s) = bernoulli(v_a));
        V_SAMPLE_PROBS(unit_type::GAUSSIAN, f(v_s) = normal_noise(v_a));

        nan_check_deep(v_a);

        if (S) {
            nan_check_deep(v_s);
        }
    }

    void activate_hidden(output_one_t& h_a, const input_one_t& input) const {
        activate_pooling<true, false>(h_a, h_a, input, input);
    }

    hidden_output_one_t hidden_features(const input_one_t& input){
        auto out = as_derived().template prepare_one_hidden_output<input_one_t>();
        activate_hidden<true, false>(out, out, input, input);
        return out;
    }

    template<typename Input>
    hidden_output_one_t hidden_features(const Input& input){
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(as_derived(), input);
        return hidden_features(converted);
    }

    weight energy(const input_one_t& v, const hidden_output_one_t& h) const {
        auto tmp = as_derived().energy_tmp();
        tmp = etl::conv_4d_valid_flipped(as_derived().reshape_v_a(v), as_derived().w);

        if (desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY) {
            //Definition according to Honglak Lee
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - c sum_v v

            return -etl::sum(as_derived().c >> etl::sum_r(v)) - etl::sum((h >> tmp(0)) + (as_derived().get_b_rep() >> h));
        } else if (desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY) {
            //Definition according to Honglak Lee / Mixed with Gaussian
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - sum_v ((v - c) ^ 2 / 2)

            auto c_rep = as_derived().get_c_rep();
            return sum(etl::pow(v - c_rep, 2) / 2.0) - etl::sum((h >> tmp(0)) + (as_derived().get_b_rep() >> h));
        } else {
            return 0.0;
        }
    }

    template<typename Input>
    weight energy(const Input& v, const hidden_output_one_t& h) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(as_derived(), v);
        return energy(converted, h);
    }

    weight free_energy(const input_one_t& v) const {
        auto tmp = as_derived().energy_tmp();
        tmp = etl::conv_4d_valid_flipped(as_derived().reshape_v_a(v), as_derived().w);

        if (desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY) {
            //Definition computed from E(v,h)

            auto b_rep = as_derived().get_b_rep();
            auto x = b_rep + tmp(0);

            return -etl::sum(as_derived().c >> etl::sum_r(v)) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else if (desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY) {
            //Definition computed from E(v,h)

            auto b_rep = as_derived().get_b_rep();
            auto x = b_rep + tmp(0);
            auto c_rep = as_derived().get_c_rep();

            return -sum(etl::pow(v - c_rep, 2) / 2.0) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else {
            return 0.0;
        }
    }

    template <typename V>
    weight free_energy(const V& v) const {
        decltype(auto) converted = converter_one<V, input_one_t>::convert(as_derived(), v);
        return free_energy(converted);
    }

    weight free_energy() const {
        return free_energy(as_derived().v1);
    }

private:
    derived_t& as_derived() {
        return *static_cast<derived_t*>(this);
    }

    const derived_t& as_derived() const {
        return *static_cast<const derived_t*>(this);
    }
};

} //end of dll namespace
