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

#include "standard_crbm_mp.hpp" //The base class

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine with Probabilistic
 * Max-Pooling.
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template <typename Desc>
struct conv_rbm_mp final : public standard_crbm_mp<conv_rbm_mp<Desc>, Desc> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = conv_rbm_mp<desc>;
    using base_type = standard_crbm_mp<this_type, desc>;

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

    conv_rbm_mp() : base_type() {
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

    size_t pool_C() const {
        return C;
    }

    auto get_b_rep() const {
        return etl::force_temporary(etl::rep<NH1, NH2>(b));
    }

    auto get_c_rep() const {
        return etl::force_temporary(etl::rep<NV1, NV2>(c));
    }

    template<typename V>
    auto get_batch_b_rep(V&& /*h*/) const {
        return etl::force_temporary(etl::rep<NH1, NH2>(b));
    }

    template<typename H>
    auto get_batch_c_rep(H&& /*h*/) const {
        static constexpr const auto batch_size = etl::decay_traits<H>::template dim<0>();
        return etl::force_temporary(etl::rep_l<batch_size>(etl::rep<NV1, NV2>(c)));
    }

    template<typename H>
    auto reshape_h_a(H&& h_a) const {
        return etl::reshape<1, K, NH1, NH2>(h_a);
    }

    template<typename V>
    auto reshape_v_a(V&& v_a) const {
        return etl::reshape<1, NC, NV1, NV2>(v_a);
    }

    auto energy_tmp() const {
        return etl::fast_dyn_matrix<weight, 1, K, NH1, NH2>();
    }

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
