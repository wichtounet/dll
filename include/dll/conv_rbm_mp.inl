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
template <typename Desc>
struct conv_rbm_mp final : public standard_conv_rbm<conv_rbm_mp<Desc>, Desc> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = conv_rbm_mp<desc>;
    using base_type = standard_conv_rbm<this_type, desc>;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit  = desc::hidden_unit;
    static constexpr const unit_type pooling_unit = desc::pooling_unit;

    static_assert(!(std::is_same<float, weight>::value && visible_unit == unit_type::GAUSSIAN),
                  "Gaussian visible units should use double-precision");

    static constexpr const std::size_t NV1 = desc::NV1; ///< The first dimension of the visible units
    static constexpr const std::size_t NV2 = desc::NV2; ///< The second dimension of the visible units
    static constexpr const std::size_t NH1 = desc::NH1; ///< The first dimension of the hidden units
    static constexpr const std::size_t NH2 = desc::NH2; ///< The second dimension of the hidden units
    static constexpr const std::size_t NC  = desc::NC;  ///< The number of input channels
    static constexpr const std::size_t K   = desc::K;   ///< The number of filters
    static constexpr const std::size_t C   = desc::C;

    static constexpr const std::size_t NW1 = NV1 - NH1 + 1; //By definition
    static constexpr const std::size_t NW2 = NV2 - NH2 + 1; //By definition
    static constexpr const std::size_t NP1 = NH1 / C;       //By definition
    static constexpr const std::size_t NP2 = NH2 / C;       //By definition

    static constexpr bool dbn_only = layer_traits<this_type>::is_dbn_only();

    using w_type = etl::fast_matrix<weight, K, NC, NW1, NW2>;
    using b_type = etl::fast_vector<weight, K>;
    using c_type = etl::fast_vector<weight, NC>;

    using input_t             = typename rbm_base_traits<this_type>::input_t;
    using output_t            = typename rbm_base_traits<this_type>::output_t;
    using input_one_t         = typename rbm_base_traits<this_type>::input_one_t;
    using output_one_t        = typename rbm_base_traits<this_type>::output_one_t;
    using hidden_output_one_t = typename rbm_base_traits<this_type>::hidden_output_one_t;

    w_type w; //!< shared weights
    b_type b; //!< hidden biases bk
    c_type c; //!< visible single bias c

    std::unique_ptr<w_type> bak_w; //!< backup shared weights
    std::unique_ptr<b_type> bak_b; //!< backup hidden biases bk
    std::unique_ptr<c_type> bak_c; //!< backup visible single bias c

    etl::fast_matrix<weight, NC, NV1, NV2> v1; ///< visible units

    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h1_a; ///< Activation probabilities of reconstructed hidden units
    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h1_s; ///< Sampled values of reconstructed hidden units

    conditional_fast_matrix_t<!dbn_only, weight, K, NP1, NP2> p1_a; ///< Activation probabilities of reconstructed hidden units
    conditional_fast_matrix_t<!dbn_only, weight, K, NP1, NP2> p1_s; ///< Sampled values of reconstructed hidden units

    conditional_fast_matrix_t<!dbn_only, weight, NC, NV1, NV2> v2_a; ///< Activation probabilities of reconstructed visible units
    conditional_fast_matrix_t<!dbn_only, weight, NC, NV1, NV2> v2_s; ///< Sampled values of reconstructed visible units

    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h2_a; ///< Activation probabilities of reconstructed hidden units
    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h2_s; ///< Sampled values of reconstructed hidden units

    conditional_fast_matrix_t<!dbn_only, weight, K, NP1, NP2> p2_a; ///< Activation probabilities of reconstructed hidden units
    conditional_fast_matrix_t<!dbn_only, weight, K, NP1, NP2> p2_s; ///< Sampled values of reconstructed hidden units

    mutable cpp::thread_pool<!layer_traits<this_type>::is_serial()> pool;

    conv_rbm_mp()
            : base_type(), pool(etl::threads) {
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        w = 0.01 * etl::normal_generator();
        b = -0.1;
        c = 0.0;
    }

    static constexpr std::size_t input_size() noexcept {
        return NV1 * NV2 * NC;
    }

    static constexpr std::size_t output_size() noexcept {
        return NP1 * NP2 * K;
    }

    static constexpr std::size_t parameters() noexcept {
        return NC * K * NW1 * NW2;
    }

    static std::string to_short_string() {
        char buffer[1024];
        snprintf(
            buffer, 1024, "CRBM_MP(%s): %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu -> %lux%lux%lu",
            to_string(hidden_unit).c_str(), NV1, NV2, NC, NW1, NW2, NH1, NH2, K, NP1, NP2, K);
        return {buffer};
    }

    // Make base class them participate in overload resolution
    using base_type::activate_hidden;

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2&) const {
        dll::auto_timer timer("crbm:mp:activate_hidden");

        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        auto b_rep = etl::force_temporary(etl::rep<NH1, NH2>(b));

        etl::reshape<1, K, NH1, NH2>(h_a) = etl::conv_4d_valid_flipped(etl::reshape<1, NC, NV1, NV2>(v_a), w);

        H_PROBS2(unit_type::BINARY, unit_type::BINARY, f(h_a) = etl::p_max_pool_h<C, C>(b_rep + h_a));
        H_PROBS2(unit_type::BINARY, unit_type::GAUSSIAN, f(h_a) = etl::p_max_pool_h<C, C>((1.0 / (0.1 * 0.1)) >> (b_rep + h_a)));
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

        etl::reshape<1, NC, NV1, NV2>(v_a) = etl::conv_4d_full(etl::reshape<1, K, NH1, NH2>(h_s), w);

        auto c_rep = etl::force_temporary(etl::rep<NV1, NV2>(c));

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

        etl::fast_dyn_matrix<weight, 1, K, NH1, NH2> v_cv; //Temporary convolution

        auto b_rep = etl::force_temporary(etl::rep<NH1, NH2>(b));

        v_cv = etl::conv_4d_valid_flipped(etl::reshape<1, NC, NV1, NV2>(v_a), w);

        if (pooling_unit == unit_type::BINARY) {
            p_a = etl::p_max_pool_p<C, C>(b_rep + v_cv(0));
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

        h_a = etl::conv_4d_valid_flipped(v_a, w);

        auto b_rep = etl::force_temporary(etl::rep<NH1, NH2>(b));

        for (size_t i = 0; i < Batch; ++i) {
            H_PROBS2(unit_type::BINARY, unit_type::BINARY, f(h_a)(i) = etl::p_max_pool_h<C, C>(b_rep + h_a(i)));
            H_PROBS2(unit_type::BINARY, unit_type::GAUSSIAN, f(h_a)(i) = etl::p_max_pool_h<C, C>((1.0 / (0.1 * 0.1)) >> (b_rep + h_a(i))));
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
    void batch_activate_visible(const H1&, const H2& h_s, V1&& v_a, V2&& v_s) const {
        dll::auto_timer timer("crbm:mp:batch_activate_visible");

        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        v_a = etl::conv_4d_full(h_s, w);

        static constexpr const auto Batch = etl::decay_traits<H1>::template dim<0>();

        auto c_rep = etl::force_temporary(etl::rep_l<Batch>(etl::rep<NV1, NV2>(c)));

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

    weight energy(const input_one_t& v, const hidden_output_one_t& h) const {
        etl::fast_dyn_matrix<weight, 1, K, NH1, NH2> tmp; //Temporary convolution
        tmp = etl::conv_4d_valid_flipped(etl::reshape<1, NC, NV1, NV2>(v), w);

        if (desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY) {
            //Definition according to Honglak Lee
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - c sum_v v

            return -etl::sum(c >> etl::sum_r(v)) - etl::sum((h >> tmp(0)) + (etl::rep<NH1, NH2>(b) >> h));
        } else if (desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY) {
            //Definition according to Honglak Lee / Mixed with Gaussian
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - sum_v ((v - c) ^ 2 / 2)

            return sum(etl::pow(v - etl::rep<NV1, NV2>(c), 2) / 2.0) - etl::sum((h >> tmp(0)) + (etl::rep<NH1, NH2>(b) >> h));
        } else {
            return 0.0;
        }
    }

    template<typename Input>
    weight energy(const Input& v, const hidden_output_one_t& h) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(*this, v);
        return energy(converted, h);
    }

    template <typename V>
    weight free_energy_impl(const V& v) const {
        etl::fast_dyn_matrix<weight, 1, K, NH1, NH2> tmp; //Temporary convolution
        tmp = etl::conv_4d_valid_flipped(etl::reshape<1, NC, NV1, NV2>(v), w);

        if (desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY) {
            //Definition computed from E(v,h)

            auto x = etl::rep<NH1, NH2>(b) + tmp(0);

            return -etl::sum(c >> etl::sum_r(v)) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else if (desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY) {
            //Definition computed from E(v,h)

            auto x = etl::rep<NH1, NH2>(b) + tmp(0);

            return -sum(etl::pow(v - etl::rep<NV1, NV2>(c), 2) / 2.0) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else {
            return 0.0;
        }
    }

    template <typename V>
    weight free_energy(const V& v) const {
        etl::fast_dyn_matrix<weight, NC, NV1, NV2> ev(v);
        return free_energy_impl(ev);
    }

    weight free_energy() const {
        return free_energy_impl(v1);
    }

    //Utilities for DBNs

    template <std::size_t B>
    using input_batch_t = etl::fast_dyn_matrix<weight, B, NC, NV1, NV2>;

    template <typename Input>
    static output_t prepare_output(std::size_t samples) {
        return output_t(samples);
    }

    template <typename Input>
    static output_one_t prepare_one_output() {
        return {};
    }

    template <typename Input>
    static hidden_output_one_t prepare_one_hidden_output() {
        return {};
    }

    hidden_output_one_t hidden_features(const input_one_t& input){
        auto out = prepare_one_hidden_output<input_one_t>();
        activate_hidden<true, false>(out, out, input, input);
        return out;
    }

    template<typename Input>
    hidden_output_one_t hidden_features(const Input& input){
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(*this, input);
        return hidden_features(converted);
    }

    void activate_hidden(output_one_t& h_a, const input_one_t& input) const {
        activate_pooling<true, false>(h_a, h_a, input, input);
    }

    template <std::size_t B>
    auto prepare_input_batch(){
        return etl::fast_dyn_matrix<weight, B, NC, NV1, NV2>();
    }

    template <std::size_t B>
    auto prepare_output_batch(){
        return etl::fast_dyn_matrix<weight, B, K, NP1, NP2>();
    }

    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(NC, NV1, NV2, K, NH1, NH2, C);
        dyn.batch_size  = layer_traits<this_type>::batch_size();
    }
};

/*!
 * \brief Simple traits to pass information around from the real
 * class to the CRTP class.
 */
template <typename Desc>
struct rbm_base_traits<conv_rbm_mp<Desc>> {
    using desc      = Desc;
    using weight    = typename desc::weight;

    using input_one_t         = etl::fast_dyn_matrix<weight, desc::NC, desc::NV1, desc::NV2>;
    using hidden_output_one_t = etl::fast_dyn_matrix<weight, desc::K, desc::NH1, desc::NH2>;
    using output_one_t        = etl::fast_dyn_matrix<weight, desc::K, desc::NH1 / desc::C, desc::NH2 / desc::C>;
    using input_t             = std::vector<input_one_t>;
    using output_t            = std::vector<output_one_t>;
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::NV1;

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::NV2;

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::NH1;

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::NH2;

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::NC;

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::NW1;

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::NW2;

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::NP1;

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::NP2;

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::K;

template <typename Desc>
const std::size_t conv_rbm_mp<Desc>::C;

} //end of dll namespace
