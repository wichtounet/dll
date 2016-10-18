//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "standard_crbm.hpp" //The base class

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template <typename Desc>
struct dyn_conv_rbm final : public standard_crbm<dyn_conv_rbm<Desc>, Desc> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = dyn_conv_rbm<desc>;
    using base_type = standard_crbm<this_type, desc>;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit  = desc::hidden_unit;

    static constexpr const bool dbn_only = layer_traits<this_type>::is_dbn_only();

    using w_type = etl::dyn_matrix<weight, 4>;
    using b_type = etl::dyn_vector<weight>;
    using c_type = etl::dyn_vector<weight>;

    using input_t      = typename rbm_base_traits<this_type>::input_t;
    using output_t     = typename rbm_base_traits<this_type>::output_t;
    using input_one_t  = typename rbm_base_traits<this_type>::input_one_t;
    using output_one_t = typename rbm_base_traits<this_type>::output_one_t;

    w_type w; //!< shared weights
    b_type b; //!< hidden biases bk
    c_type c; //!< visible single bias c

    std::unique_ptr<w_type> bak_w; //!< backup shared weights
    std::unique_ptr<b_type> bak_b; //!< backup hidden biases bk
    std::unique_ptr<c_type> bak_c; //!< backup visible single bias c

    etl::dyn_matrix<weight, 3> v1; ///< visible units

    etl::dyn_matrix<weight, 3> h1_a; ///< Activation probabilities of reconstructed hidden units
    etl::dyn_matrix<weight, 3> h1_s; ///< Sampled values of reconstructed hidden units

    etl::dyn_matrix<weight, 3> v2_a; ///< Activation probabilities of reconstructed visible units
    etl::dyn_matrix<weight, 3> v2_s; ///< Sampled values of reconstructed visible units

    etl::dyn_matrix<weight, 3> h2_a; ///< Activation probabilities of reconstructed hidden units
    etl::dyn_matrix<weight, 3> h2_s; ///< Sampled values of reconstructed hidden units

    size_t nv1; ///< The first visible dimension
    size_t nv2; ///< The second visible dimension
    size_t nh1; ///< The first output dimension
    size_t nh2; ///< The second output dimension
    size_t nc;  ///< The number of input channels
    size_t k;   ///< The number of filters

    size_t nw1; ///< The first dimension of the filters
    size_t nw2; ///< The second dimension of the filters

    size_t batch_size = 25;

    dyn_conv_rbm() : base_type() {
        // Nothing else to init
    }

    void prepare_input(input_one_t& input) const {
        input = input_one_t(nc, nv1, nv2);
    }

    void init_layer(size_t nc, size_t nv1, size_t nv2, size_t k, size_t nh1, size_t nh2){
        this->nv1 = nv1;
        this->nv2 = nv2;
        this->nh1 = nh1;
        this->nh2 = nh2;
        this->nc = nc;
        this->k = k;

        this->nw1 = nv1 - nh1 + 1;
        this->nw2 = nv2 - nh2 + 1;

        w = etl::dyn_matrix<weight, 4>(k, nc, nw1, nw2);

        b = etl::dyn_vector<weight>(k);
        c = etl::dyn_vector<weight>(nc);

        v1 = etl::dyn_matrix<weight, 3>(nc, nv1, nv2);

        h1_a = etl::dyn_matrix<weight, 3>(k, nh1, nh2);
        h1_s = etl::dyn_matrix<weight, 3>(k, nh1, nh2);

        v2_a = etl::dyn_matrix<weight, 3>(nc, nv1, nv2);
        v2_s = etl::dyn_matrix<weight, 3>(nc, nv1, nv2);

        h2_a = etl::dyn_matrix<weight, 3>(k, nh1, nh2);
        h2_s = etl::dyn_matrix<weight, 3>(k, nh1, nh2);

        if (is_relu(hidden_unit)) {
            w = etl::normal_generator(0.0, 0.01);
            b = 0.0;
            c = 0.0;
        } else {
            w = 0.01 * etl::normal_generator();
            b = -0.1;
            c = 0.0;
        }
    }

    std::size_t input_size() const noexcept {
        return nv1 * nv2 * nc;
    }

    std::size_t output_size() const noexcept {
        return nh1 * nh2 * k;
    }

    std::size_t parameters() const noexcept {
        return nc * k * nw1 * nw2;
    }

    std::string to_short_string() const {
        char buffer[1024];
        snprintf(
            buffer, 1024, "CRBM(dyn)(%s): %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu",
            to_string(hidden_unit).c_str(), nv1, nv2, nc, nw1, nw2, nh1, nh2, k);
        return {buffer};
    }

    template <typename Input>
    output_t prepare_output(std::size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(size_t i = 0; i < samples; ++i){
            output.emplace_back(k, nh1, nh2);
        }
        return output;
    }

    template <typename Input>
    output_one_t prepare_one_output() const {
        return output_one_t(k, nh1, nh2);
    }

    template <std::size_t B>
    auto prepare_input_batch() const {
        return etl::dyn_matrix<weight, 4>(B, nc, nv1, nv2);
    }

    template <std::size_t B>
    auto prepare_output_batch() const {
        return etl::dyn_matrix<weight, 4>(B, k, nh1, nh2);
    }

    template <typename DBN>
    void init_sgd_context() {
        this->sgd_context_ptr = std::make_shared<sgd_context<DBN, this_type>>(nc, nv1, nv2, k, nh1, nh2);
    }

    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
    }

    friend base_type;

private:
    auto get_b_rep() const {
        return etl::force_temporary(etl::rep(b, nh1, nh2));
    }

    auto get_c_rep() const {
        return etl::force_temporary(etl::rep(c, nv1, nv2));
    }

    template<typename V>
    auto get_batch_b_rep(V&& v) const {
        const auto batch_size = etl::dim<0>(v);
        return etl::force_temporary(etl::rep_l(etl::rep(b, nh1, nh2), batch_size));
    }

    template<typename H>
    auto get_batch_c_rep(H&& h) const {
        const auto batch_size = etl::dim<0>(h);
        return etl::force_temporary(etl::rep_l(etl::rep(c, nv1, nv2), batch_size));
    }

    template<typename H>
    auto reshape_h_a(H&& h_a) const {
        return etl::reshape(h_a, 1, k, nh1, nh2);
    }

    template<typename V>
    auto reshape_v_a(V&& v_a) const {
        return etl::reshape(v_a, 1, nc, nv1, nv2);
    }

    auto energy_tmp() const {
        return etl::dyn_matrix<weight, 4>(1UL, k, nh1, nh2);
    }

    template <typename V1, typename V2, std::size_t Off = 0>
    static void validate_inputs() {
        static_assert(etl::decay_traits<V1>::dimensions() == 3 + Off, "Inputs must be 3D");
        static_assert(etl::decay_traits<V2>::dimensions() == 3 + Off, "Inputs must be 3D");
    }

    template <typename H1, typename H2, std::size_t Off = 0>
    static void validate_outputs() {
        static_assert(etl::decay_traits<H1>::dimensions() == 3 + Off, "Outputs must be 3D");
        static_assert(etl::decay_traits<H2>::dimensions() == 3 + Off, "Outputs must be 3D");
    }
};

/*!
 * \brief Simple traits to pass information around from the real
 * class to the CRTP class.
 */
template <typename Desc>
struct rbm_base_traits<dyn_conv_rbm<Desc>> {
    using desc      = Desc;
    using weight    = typename desc::weight;

    using input_one_t         = etl::dyn_matrix<weight, 3>;
    using output_one_t        = etl::dyn_matrix<weight, 3>;
    using hidden_output_one_t = output_one_t;
    using input_t             = std::vector<input_one_t>;
    using output_t            = std::vector<output_one_t>;
};

} //end of dll namespace
