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

#include "dll/base_traits.hpp"
#include "standard_crbm_mp.hpp" //The base class

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine with Probabilistic
 * Max-Pooling.
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template <typename Desc>
struct conv_rbm_mp_impl final : public standard_crbm_mp<conv_rbm_mp_impl<Desc>, Desc> {
    using desc      = Desc; ///< The descriptor of the layer
    using weight    = typename desc::weight; ///< The data type for this layer
    using this_type = conv_rbm_mp_impl<desc>; ///< The type of this layer
    using base_type = standard_crbm_mp<this_type, desc>;
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    static constexpr unit_type visible_unit = desc::visible_unit; ///< The type of visible unit
    static constexpr unit_type hidden_unit  = desc::hidden_unit;  ///< The type of hidden unit
    static constexpr unit_type pooling_unit = desc::pooling_unit; ///< The type of pooling unit

    static inline constexpr size_t NV1 = desc::NV1; ///< The first dimension of the visible units
    static inline constexpr size_t NV2 = desc::NV2; ///< The second dimension of the visible units
    static inline constexpr size_t NW1 = desc::NW1; ///< The first dimension of the hidden units
    static inline constexpr size_t NW2 = desc::NW2; ///< The second dimension of the hidden units
    static inline constexpr size_t NC  = desc::NC;  ///< The number of input channels
    static inline constexpr size_t K   = desc::K;   ///< The number of filters
    static inline constexpr size_t C   = desc::C;

    static inline constexpr size_t NH1 = NV1 - NW1 + 1; //By definition
    static inline constexpr size_t NH2 = NV2 - NW2 + 1; //By definition
    static inline constexpr size_t NP1 = NH1 / C;       //By definition
    static inline constexpr size_t NP2 = NH2 / C;       //By definition

    static constexpr size_t batch_size  = desc::BatchSize;  ///< The mini-batch size

    static constexpr bool dbn_only = rbm_layer_traits<this_type>::is_dbn_only();

    using w_type = etl::fast_matrix<weight, K, NC, NW1, NW2>; ///< The type of the weights
    using b_type = etl::fast_vector<weight, K>; ///< The type of the biases
    using c_type = etl::fast_vector<weight, NC>;

    using input_t             = typename rbm_base_traits<this_type>::input_t; ///< The type of the input
    using output_t            = typename rbm_base_traits<this_type>::output_t; ///< The type of the output
    using input_one_t         = typename rbm_base_traits<this_type>::input_one_t; ///< The type of one input
    using output_one_t        = typename rbm_base_traits<this_type>::output_one_t; ///< The type of one output
    using hidden_output_one_t = typename rbm_base_traits<this_type>::hidden_output_one_t;

    w_type w; ///< shared weights
    b_type b; ///< hidden biases bk
    c_type c; ///< visible single bias c

    std::unique_ptr<w_type> bak_w; ///< backup shared weights
    std::unique_ptr<b_type> bak_b; ///< backup hidden biases bk
    std::unique_ptr<c_type> bak_c; ///< backup visible single bias c

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

    conv_rbm_mp_impl() : base_type() {
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        w = 0.01 * etl::normal_generator();
        b = -0.1;
        c = 0.0;
    }

    /*!
     * \brief Return the input size of the layer
     */
    static constexpr size_t input_size() noexcept {
        return NV1 * NV2 * NC;
    }

    /*!
     * \brief Return the output size of the layer
     */
    static constexpr size_t output_size() noexcept {
        return NP1 * NP2 * K;
    }

    /*!
     * \brief Return the number of parameters of the layer
     */
    static constexpr size_t parameters() noexcept {
        return NC * K * NW1 * NW2;
    }

    /*!
     * \brief Return a textual representation of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        char buffer[1024];
        snprintf(
            buffer, 1024, "CRBM_MP(%s->%s)",
            to_string(visible_unit).c_str(), to_string(hidden_unit).c_str());
        return {buffer};
    }

    /*!
     * \brief Return a textual representation of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        char buffer[1024];
        snprintf(
            buffer, 1024, "CRBM_MP(%s->%s): %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu -> %lux%lux%lu",
            to_string(visible_unit).c_str(), to_string(hidden_unit).c_str(), NV1, NV2, NC, NW1, NW2, NH1, NH2, K, NP1, NP2, K);
        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {K, NP1, NP2};
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    static output_t prepare_output(size_t samples) {
        return output_t(samples);
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    static output_one_t prepare_one_output() {
        return {};
    }

    template <typename Input>
    static hidden_output_one_t prepare_one_hidden_output() {
        return {};
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(NC, NV1, NV2, K, NW1, NW2, C);
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
        return C;
    }

    auto get_b_rep() const {
        return etl::force_temporary(etl::rep<NH1, NH2>(b));
    }

    auto get_c_rep() const {
        return etl::force_temporary(etl::rep<NV1, NV2>(c));
    }

    template<typename V>
    auto get_batch_b_rep(V&& v) const {
        if constexpr (etl::all_fast<V>) {
            static constexpr auto batch_size = etl::decay_traits<V>::template dim<0>();
            return etl::force_temporary(etl::rep_l<batch_size>(etl::rep<NH1, NH2>(b)));
        } else {
            const auto batch_size = etl::dim<0>(v);
            return etl::force_temporary(etl::rep_l(etl::rep<NH1, NH2>(b), batch_size));
        }
    }

    template<typename H>
    auto get_batch_c_rep(H&& /*h*/) const {
        static constexpr auto batch_size = etl::decay_traits<H>::template dim<0>();
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
};

/*!
 * \brief Simple traits to pass information around from the real
 * class to the CRTP class.
 */
template <typename Desc>
struct rbm_base_traits<conv_rbm_mp_impl<Desc>> {
    using desc      = Desc; ///< The descriptor of the layer
    using weight    = typename desc::weight; ///< The data type for this layer

    using input_one_t         = etl::fast_dyn_matrix<weight, desc::NC, desc::NV1, desc::NV2>; ///< The type of one input
    using hidden_output_one_t = etl::fast_dyn_matrix<weight, desc::K, desc::NV1 - desc::NW1 + 1, desc::NV2 - desc::NW2 + 1>;
    using output_one_t        = etl::fast_dyn_matrix<weight, desc::K, (desc::NV1 - desc::NW1 + 1) / desc::C, (desc::NV2 - desc::NW2 + 1) / desc::C>; ///< The type of one output
    using input_t             = std::vector<input_one_t>; ///< The type of the input
    using output_t            = std::vector<output_one_t>; ///< The type of the output
};

// Declare the traits for the RBM

template<typename Desc>
struct layer_base_traits<conv_rbm_mp_impl<Desc>> {
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
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = Desc::hidden_unit != dll::unit_type::SOFTMAX; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = false;  ///< Indicates if the layer is supported by SGD
};

template<typename Desc>
struct rbm_layer_base_traits<conv_rbm_mp_impl<Desc>> {
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
