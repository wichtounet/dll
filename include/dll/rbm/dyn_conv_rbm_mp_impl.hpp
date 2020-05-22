//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "standard_crbm_mp.hpp" //The base class

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template <typename Desc>
struct dyn_conv_rbm_mp_impl final : public standard_crbm_mp<dyn_conv_rbm_mp_impl<Desc>, Desc> {
    using desc      = Desc; ///< The descriptor of the layer
    using weight    = typename desc::weight; ///< The data type for this layer
    using this_type = dyn_conv_rbm_mp_impl<desc>; ///< The type of this layer
    using base_type = standard_crbm_mp<this_type, desc>;
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    static constexpr unit_type visible_unit = desc::visible_unit;
    static constexpr unit_type hidden_unit  = desc::hidden_unit;
    static constexpr unit_type pooling_unit = desc::pooling_unit;
    static constexpr size_t batch_size      = desc::BatchSize; ///< The mini-batch size

    static constexpr bool dbn_only = rbm_layer_traits<this_type>::is_dbn_only();

    using w_type = etl::dyn_matrix<weight, 4>; ///< The type of the weights
    using b_type = etl::dyn_vector<weight>; ///< The type of the biases
    using c_type = etl::dyn_vector<weight>;

    using input_t      = typename rbm_base_traits<this_type>::input_t; ///< The type of the input
    using output_t     = typename rbm_base_traits<this_type>::output_t; ///< The type of the output
    using input_one_t  = typename rbm_base_traits<this_type>::input_one_t; ///< The type of one input
    using output_one_t = typename rbm_base_traits<this_type>::output_one_t; ///< The type of one output
    using hidden_output_one_t = typename rbm_base_traits<this_type>::hidden_output_one_t;

    w_type w; ///< shared weights
    b_type b; ///< hidden biases bk
    c_type c; ///< visible single bias c

    std::unique_ptr<w_type> bak_w; ///< backup shared weights
    std::unique_ptr<b_type> bak_b; ///< backup hidden biases bk
    std::unique_ptr<c_type> bak_c; ///< backup visible single bias c

    etl::dyn_matrix<weight, 3> v1; //visible units

    etl::dyn_matrix<weight, 3> h1_a; ///< Activation probabilities of reconstructed hidden units
    etl::dyn_matrix<weight, 3> h1_s; ///< Sampled values of reconstructed hidden units

    etl::dyn_matrix<weight, 3> p1_a; ///< Pooled Activation probabilities of reconstructed hidden units
    etl::dyn_matrix<weight, 3> p1_s; ///< Pooled Sampled values of reconstructed hidden units

    etl::dyn_matrix<weight, 3> v2_a; ///< Activation probabilities of reconstructed visible units
    etl::dyn_matrix<weight, 3> v2_s; ///< Sampled values of reconstructed visible units

    etl::dyn_matrix<weight, 3> h2_a; ///< Activation probabilities of reconstructed hidden units
    etl::dyn_matrix<weight, 3> h2_s; ///< Sampled values of reconstructed hidden units

    etl::dyn_matrix<weight, 3> p2_a; ///< Pooled Activation probabilities of reconstructed hidden units
    etl::dyn_matrix<weight, 3> p2_s; ///< Pooled Sampled values of reconstructed hidden units

    size_t nv1; ///< The first visible dimension
    size_t nv2; ///< The second visible dimension
    size_t nh1; ///< The first output dimension
    size_t nh2; ///< The second output dimension
    size_t nc;  ///< The number of input channels
    size_t k;   ///< The number of filters
    size_t p_c;

    size_t nw1; ///< The first dimension of the filters
    size_t nw2; ///< The second dimension of the filters
    size_t np1;
    size_t np2;

    dyn_conv_rbm_mp_impl() : base_type() {
        // Nothing else to init
    }

    void prepare_input(input_one_t& input) const {
        input = input_one_t(nc, nv1, nv2);
    }

    /*!
     * \brief Initialize the dynamic layer
     */
    void init_layer(size_t nc, size_t nv1, size_t nv2, size_t k, size_t nw1, size_t nw2, size_t p_c){
        this->nv1 = nv1;
        this->nv2 = nv2;
        this->nw1 = nw1;
        this->nw2 = nw2;
        this->nc = nc;
        this->k = k;
        this->p_c = p_c;

        this->nh1 = nv1 - nw1 + 1;
        this->nh2 = nv2 - nw2 + 1;

        this->np1 = nh1 / p_c;
        this->np2 = nh2 / p_c;

        w = etl::dyn_matrix<weight, 4>(k, nc, nw1, nw2);
        b = etl::dyn_vector<weight>(k);
        c = etl::dyn_vector<weight>(nc);

        v1 = etl::dyn_matrix<weight, 3>(nc, nv1, nv2);

        h1_a = etl::dyn_matrix<weight, 3>(k, nh1, nh2);
        h1_s = etl::dyn_matrix<weight, 3>(k, nh1, nh2);

        p1_a = etl::dyn_matrix<weight, 3>(k, np1, np2);
        p1_s = etl::dyn_matrix<weight, 3>(k, np1, np2);

        v2_a = etl::dyn_matrix<weight, 3>(nc, nv1, nv2);
        v2_s = etl::dyn_matrix<weight, 3>(nc, nv1, nv2);

        h2_a = etl::dyn_matrix<weight, 3>(k, nh1, nh2);
        h2_s = etl::dyn_matrix<weight, 3>(k, nh1, nh2);

        p2_a = etl::dyn_matrix<weight, 3>(k, np1, np2);
        p2_s = etl::dyn_matrix<weight, 3>(k, np1, np2);

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

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    size_t input_size() const noexcept {
        return nv1 * nv2 * nc;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    size_t output_size() const noexcept {
        return np1 * np2 * k;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    size_t parameters() const noexcept {
        return nc * k * nw1 * nw2;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_short_string([[maybe_unused]] std::string pre = "") const {
        char buffer[1024];
        snprintf(
            buffer, 1024, "CRBM_MP(%s) (dyn)",
            to_string(hidden_unit).c_str());
        return {buffer};
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_full_string([[maybe_unused]] std::string pre = "") const {
        char buffer[1024];
        snprintf(
            buffer, 1024, "CRBM_MP(dyn)(%s): %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu -> %lux%lux%lu",
            to_string(hidden_unit).c_str(), nv1, nv2, nc, nw1, nw2, nh1, nh2, k, np1, np2, k);
        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {k, np1, np2};
    }

    //Utilities for DBNs

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
            output.emplace_back(k, np1, np2);
        }
        return output;
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    output_one_t prepare_one_output() const {
        return output_one_t(k, np1, np2);
    }

    template <typename Input>
    output_one_t prepare_one_hidden_output() const {
        return output_one_t(k, nh1, nh2);
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void forward_batch(Output& output, const Input& input) const {
        this->batch_activate_pooling(output, input);
    }

    friend base_type;

private:
    size_t pool_C() const {
        return p_c;
    }

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

};

/*!
 * \brief Simple traits to pass information around from the real
 * class to the CRTP class.
 */
template <typename Desc>
struct rbm_base_traits<dyn_conv_rbm_mp_impl<Desc>> {
    using desc      = Desc; ///< The descriptor of the layer
    using weight    = typename desc::weight; ///< The data type for this layer

    using input_one_t         = etl::dyn_matrix<weight, 3>; ///< The type of one input
    using output_one_t        = etl::dyn_matrix<weight, 3>; ///< The type of one output
    using hidden_output_one_t = etl::dyn_matrix<weight, 3>;
    using input_t             = std::vector<input_one_t>; ///< The type of the input
    using output_t            = std::vector<output_one_t>; ///< The type of the output
};

// Declare the traits for the RBM

template<typename Desc>
struct layer_base_traits<dyn_conv_rbm_mp_impl<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = true;  ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = true;  ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = true;  ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = Desc::hidden_unit != dll::unit_type::SOFTMAX; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = false;  ///< Indicates if the layer is supported by SGD
};

template<typename Desc>
struct rbm_layer_base_traits<dyn_conv_rbm_mp_impl<Desc>> {
    using param = typename Desc::parameters;

    static constexpr bool has_momentum       = param::template contains<momentum>();                       ///< Does the RBM has momentum
    static constexpr bool has_clip_gradients = param::template contains<clip_gradients>();                 ///< Does the RBM has gradient clipping
    static constexpr bool is_verbose         = param::template contains<verbose>();                        ///< Does the RBM is verbose
    static constexpr bool has_shuffle        = param::template contains<shuffle>();                        ///< Does the RBM has shuffle
    static constexpr bool is_dbn_only        = param::template contains<dbn_only>();                       ///< Does the RBM is only used inside a DBN
    static constexpr bool has_init_weights   = param::template contains<init_weights>();                   ///< Does the RBM use weights initialization
    static constexpr bool has_free_energy    = param::template contains<free_energy>();                    ///< Does the RBM displays the free energy
    static constexpr auto sparsity_method    = get_value_l_v<sparsity<dll::sparsity_method::NONE>, param>; ///< The RBM's sparsity method
    static constexpr auto bias_mode          = get_value_l_v<bias<dll::bias_mode::NONE>, param>;           ///< The RBM's sparsity bias mode
    static constexpr auto decay              = get_value_l_v<weight_decay<dll::decay_type::NONE>, param>;  ///< The RMB's sparsity decay type
    static constexpr bool has_sparsity       = sparsity_method != dll::sparsity_method::NONE;              ///< Does the RBM has sparsity
};

} //end of dll namespace
