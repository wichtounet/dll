//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

#include "dll/rbm/standard_conv_rbm.hpp" //The base class
#include "dll/base_conf.hpp"             //The configuration helpers
#include "dll/rbm/rbm_tmp.hpp"           // static_if macros

namespace dll {

// TODO: This should be retested and rechecked, likely not working

/*!
 * \brief Convolutional Restricted Boltzmann Machine with Probabilistic
 * Max-Pooling.
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template <typename Derived, typename Desc>
struct standard_crbm_mp : public standard_conv_rbm<Derived, Desc> {
    using derived_t = Derived;                           ///< The derived type
    using desc      = Desc;                              ///< The descriptor of the layer
    using weight    = typename desc::weight;             ///< The data type for this layer
    using this_type = standard_crbm_mp<derived_t, desc>; ///< The type of this layer
    using base_type = standard_conv_rbm<Derived, desc>;  ///< The base type
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    using input_one_t         = typename rbm_base_traits<derived_t>::input_one_t;         ///< The type of one input
    using output_one_t        = typename rbm_base_traits<derived_t>::output_one_t;        ///< The type of one output
    using input_t             = typename rbm_base_traits<derived_t>::input_t;             ///< The type of the input
    using output_t            = typename rbm_base_traits<derived_t>::output_t;            ///< The type of the output
    using hidden_output_one_t = typename rbm_base_traits<derived_t>::hidden_output_one_t; ///< The type of one output (hidden, not pooling)

    static constexpr unit_type visible_unit = desc::visible_unit; ///< The visible unit type
    static constexpr unit_type hidden_unit  = desc::hidden_unit;  ///< The hidden unit type
    static constexpr unit_type pooling_unit = desc::pooling_unit; ///< The pooling unit type

    standard_crbm_mp() = default;

    // Make base class them participate in overload resolution
    using base_type::activate_hidden;
    using base_type::batch_activate_hidden;

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2&) const {
        dll::auto_timer timer("crbm:mp:activate_hidden");

        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        auto b_rep = as_derived().get_b_rep();

        as_derived().reshape_h_a(h_a) = etl::conv_4d_valid_flipped(as_derived().reshape_v_a(v_a), as_derived().w);

        // TODO: This is a huge mess
        // Note: this is wrong because of PMP

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
            h_a = etl::p_max_pool_h(b_rep + h_a, this->C(), this->C());
        }

        if constexpr (P && hidden_unit == unit_type::BINARY && visible_unit == unit_type::GAUSSIAN) {
            h_a = etl::p_max_pool_h((1.0 / (0.1 * 0.1)) >> (b_rep + h_a), this->C(), this->C());
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

        // Nan Checks

        if (P) {
            nan_check_etl(h_a);
        }

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

        // Compute the visible activation probabilities

        if constexpr (P && visible_unit == unit_type::BINARY) {
            v_a = etl::sigmoid(c_rep + v_a);
        }

        if constexpr (P && visible_unit == unit_type::GAUSSIAN) {
            v_a = c_rep + v_a;
        }

        // Sample the visible values

        if constexpr (P && S && visible_unit == unit_type::BINARY) {
            v_s = bernoulli(v_a);
        }

        if constexpr (P && S && visible_unit == unit_type::GAUSSIAN) {
            v_s = normal_noise(v_a);
        }

        // Nan Checks

        if (P) {
            nan_check_deep(v_a);
        }

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

        // Compute the pooling activation probabilities

        if constexpr (P && pooling_unit == unit_type::BINARY) {
            p_a = etl::p_max_pool_p(b_rep + v_cv(0), C(), C());
        }

        // Sample the pooling values

        if constexpr (S && pooling_unit == unit_type::BINARY) {
            p_s = r_bernoulli(p_a);
        }

        // Nan Checks

        if (P) {
            nan_check_etl(p_a);
        }

        if (S) {
            nan_check_etl(p_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void batch_activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2&) const {
        dll::auto_timer timer("crbm:mp:batch_activate_hidden");

        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        [[maybe_unused]] const auto Batch = etl::dim<0>(h_a);

        cpp_assert(etl::dim<0>(h_s) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");

        h_a = etl::conv_4d_valid_flipped(v_a, as_derived().w);

        auto b_rep = as_derived().get_batch_b_rep(v_a);

        // TODO: This is a huge mess

        // Note: this is wrong because of PMP

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
            h_a = etl::p_max_pool_h(b_rep + h_a, this->C(), this->C());
        }

        if constexpr (P && hidden_unit == unit_type::BINARY && visible_unit == unit_type::GAUSSIAN) {
            h_a = etl::p_max_pool_h((1.0 / (0.1 * 0.1)) >> (b_rep + h_a), this->C(), this->C());
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

        // Nan Checks

        if (P) {
            nan_check_deep(h_a);
        }

        if (S) {
            nan_check_deep(h_s);
        }
    }

    template <bool P = true, bool S = true, typename Po, typename V>
    void batch_activate_pooling(Po& p_a, Po& p_s, const V& v_a, const V&) const {
        dll::auto_timer timer("crbm:mp:activate_pooling");

        static_assert(pooling_unit == unit_type::BINARY, "Invalid pooling unit type");
        static_assert(P, "Computing S without P is not implemented");

        [[maybe_unused]] const auto Batch = etl::dim<0>(p_a);

        cpp_assert(etl::dim<0>(p_s) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");

        auto b_rep = as_derived().get_batch_b_rep(v_a);

        auto h_a = etl::force_temporary(etl::conv_4d_valid_flipped(v_a, as_derived().w));

        // Compute the pooling activation probabilities

        if constexpr (P && pooling_unit == unit_type::BINARY) {
            p_a = etl::p_max_pool_p(b_rep + h_a, C(), C());
        }

        // Sample the pooling values

        if constexpr (S && pooling_unit == unit_type::BINARY) {
            p_s = r_bernoulli(p_a);
        }

        // Nan Checks

        if (P) {
            nan_check_etl(p_a);
        }

        if (S) {
            nan_check_etl(p_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void batch_activate_visible(const H1& h_a, const H2& h_s, V1&& v_a, V2&& v_s) const {
        dll::auto_timer timer("crbm:mp:batch_activate_visible");

        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        v_a = etl::conv_4d_full(h_s, as_derived().w);

        auto c_rep = as_derived().get_batch_c_rep(h_s);

        [[maybe_unused]] const auto Batch = etl::dim<0>(h_a);

        cpp_assert(etl::dim<0>(h_s) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_s) == Batch, "The number of batch must be consistent");

        // Compute the visible activation probabilities

        if constexpr (P && visible_unit == unit_type::BINARY) {
            v_a = etl::sigmoid(c_rep + v_a);
        }

        if constexpr (P && visible_unit == unit_type::GAUSSIAN) {
            v_a = c_rep + v_a;
        }

        // Sample the visible values

        if constexpr (P && S && visible_unit == unit_type::BINARY) {
            v_s = bernoulli(v_a);
        }

        if constexpr (P && S && visible_unit == unit_type::GAUSSIAN) {
            v_s = normal_noise(v_a);
        }

        // Nan Checks

        if (P) {
            nan_check_deep(v_a);
        }

        if (S) {
            nan_check_deep(v_s);
        }
    }

    template<typename Input>
    void activate_hidden(output_one_t& h_a, const Input& input) const {
        activate_pooling<true, false>(h_a, h_a, input, input);
    }

    template <typename Po, typename V>
    void batch_activate_pooling(Po&& p_a, const V& v_a) const {
        batch_activate_pooling<true, false>(p_a, p_a, v_a, v_a);
    }

    template<typename Input>
    hidden_output_one_t hidden_features(const Input& input){
        auto out = as_derived().template prepare_one_hidden_output<input_one_t>();
        activate_hidden<true, false>(out, out, input, input);
        return out;
    }

    // It is necessary to use here in order to pool

    /*!
     * \brief Compute the test presentation for a given input
     * \param output The output to fill
     * \param input The input to compute the representation from
     */
    template <typename Input, typename Output>
    void test_activate_hidden(Output&& output, const Input& input) const {
        batch_activate_pooling(batch_reshape(output), batch_reshape(input));
    }

    /*!
     * \brief Compute the train presentation for a given input
     * \param output The output to fill
     * \param input The input to compute the representation from
     */
    template <typename Input, typename Output>
    void train_activate_hidden(Output&& output, const Input& input) const {
        batch_activate_pooling(batch_reshape(output), batch_reshape(input));
    }

    template <bool Train, typename Input, typename Output>
    void select_activate_hidden(Output&& output, const Input& input) const {
        batch_activate_pooling(batch_reshape(output), batch_reshape(input));
    }

    friend base_type;

private:
    size_t C() const {
        return as_derived().pool_C();
    }

    template<typename Input, typename Out>
    weight energy_impl(const Input& v, const Out& h) const {
        static_assert(etl::is_etl_expr<Out>, "energy_impl works with ETL expressions only");

        auto rv = as_derived().reshape_v_a(v);
        auto tmp = as_derived().energy_tmp();
        tmp = etl::conv_4d_valid_flipped(as_derived().reshape_v_a(rv), as_derived().w);

        if  constexpr (desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY) {
            //Definition according to Honglak Lee
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - c sum_v v

            return -etl::sum(as_derived().c >> etl::sum_r(rv(0))) - etl::sum((h >> tmp(0)) + (as_derived().get_b_rep() >> h));
        } else if  constexpr (desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY) {
            //Definition according to Honglak Lee / Mixed with Gaussian
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - sum_v ((v - c) ^ 2 / 2)

            auto c_rep = as_derived().get_c_rep();
            return sum(etl::pow(rv(0) - c_rep, 2) / 2.0) - etl::sum((h >> tmp(0)) + (as_derived().get_b_rep() >> h));
        } else {
            return 0.0;
        }
    }

    template <typename Input>
    weight free_energy_impl(const Input& v) const {
        auto rv = as_derived().reshape_v_a(v);
        auto tmp = as_derived().energy_tmp();
        tmp = etl::conv_4d_valid_flipped(rv, as_derived().w);

        if  constexpr (desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY) {
            //Definition computed from E(v,h)

            auto b_rep = as_derived().get_b_rep();
            auto x = b_rep + tmp(0);

            return -etl::sum(as_derived().c >> etl::sum_r(rv(0))) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else if  constexpr (desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY) {
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
