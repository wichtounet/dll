//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementation of a Convolutional Restricted Boltzmann Machine
 */

#pragma once

#include "dll/base_traits.hpp"
#include "dll/rbm/standard_crbm.hpp" //The base class

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template <typename Desc>
struct conv_rbm final : public standard_crbm<conv_rbm<Desc>, Desc> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = conv_rbm<desc>;
    using base_type = standard_crbm<this_type, desc>;

    static constexpr unit_type visible_unit = desc::visible_unit; ///< The type of visible unit
    static constexpr unit_type hidden_unit  = desc::hidden_unit;  ///< The type of hidden unit

    static constexpr size_t NV1 = desc::NV1; ///< The first dimension of the visible units
    static constexpr size_t NV2 = desc::NV2; ///< The second dimension of the visible units
    static constexpr size_t NW1 = desc::NW1; ///< The first dimension of the hidden units
    static constexpr size_t NW2 = desc::NW2; ///< The second dimension of the hidden units
    static constexpr size_t NC  = desc::NC;  ///< The number of input channels
    static constexpr size_t K   = desc::K;   ///< The number of filters

    static constexpr size_t batch_size  = desc::BatchSize;  ///< The mini-batch size

    static constexpr size_t NH1 = NV1 - NW1 + 1; //By definition
    static constexpr size_t NH2 = NV2 - NW2 + 1; //By definition

    static constexpr bool dbn_only = rbm_layer_traits<this_type>::is_dbn_only();

    using w_type = etl::fast_matrix<weight, K, NC, NW1, NW2>;
    using b_type = etl::fast_vector<weight, K>;
    using c_type = etl::fast_vector<weight, NC>;

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

    etl::fast_matrix<weight, NC, NV1, NV2> v1; //visible units

    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h1_a; ///< Activation probabilities of reconstructed hidden units
    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h1_s; ///< Sampled values of reconstructed hidden units

    conditional_fast_matrix_t<!dbn_only, weight, NC, NV1, NV2> v2_a; ///< Activation probabilities of reconstructed visible units
    conditional_fast_matrix_t<!dbn_only, weight, NC, NV1, NV2> v2_s; ///< Sampled values of reconstructed visible units

    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h2_a; ///< Activation probabilities of reconstructed hidden units
    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h2_s; ///< Sampled values of reconstructed hidden units

    conv_rbm() : base_type() {
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
     * \brief Return the input size of the layer
     */
    static constexpr size_t input_size() noexcept {
        return NV1 * NV2 * NC;
    }

    /*!
     * \brief Return the output size of the layer
     */
    static constexpr size_t output_size() noexcept {
        return NH1 * NH2 * K;
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
    static std::string to_short_string() {
        char buffer[1024];
        snprintf(
            buffer, 1024, "CRBM(%s->%s): %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu",
            to_string(visible_unit).c_str(), to_string(hidden_unit).c_str(), NV1, NV2, NC, NW1, NW2, NH1, NH2, K);
        return {buffer};
    }

    template <typename Input>
    static output_t prepare_output(size_t samples) {
        return output_t{samples};
    }

    template <typename Input>
    static output_one_t prepare_one_output() {
        return output_one_t{};
    }

    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(NC, NV1, NV2, K, NW1, NW2);
        dyn.batch_size  = batch_size;
    }

    /*!
     * \brief Adapt the errors, called before backpropagation of the errors.
     *
     * This must be used by layers that have both an activation fnction and a non-linearity.
     *
     * \param context the training context
     */
    template<typename C>
    void adapt_errors(C& context) const {
        static_assert(
            hidden_unit == unit_type::BINARY || hidden_unit == unit_type::RELU || hidden_unit == unit_type::SOFTMAX,
            "Only (C)RBM with binary, softmax or RELU hidden unit are supported");

        static constexpr function activation_function =
            hidden_unit == unit_type::BINARY
                ? function::SIGMOID
                : (hidden_unit == unit_type::SOFTMAX ? function::SOFTMAX : function::RELU);

        context.errors = f_derivative<activation_function>(context.output) >> context.errors;
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        output = etl::conv_4d_full_flipped(context.errors, w);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        context.w_grad = conv_4d_valid_filter_flipped(context.input, context.errors);
        context.b_grad = etl::mean_r(etl::sum_l(context.errors));
    }

    friend base_type;

private:
    auto get_b_rep() const {
        return etl::force_temporary(etl::rep<NH1, NH2>(b));
    }

    auto get_c_rep() const {
        return etl::force_temporary(etl::rep<NV1, NV2>(c));
    }

    template<typename V, cpp_enable_if(etl::all_fast<V>::value)>
    auto get_batch_b_rep(V&& /*h*/) const {
        static constexpr auto batch_size = etl::decay_traits<V>::template dim<0>();
        return etl::force_temporary(etl::rep_l<batch_size>(etl::rep<NH1, NH2>(b)));
    }

    template<typename V, cpp_disable_if(etl::all_fast<V>::value)>
    auto get_batch_b_rep(V&& v) const {
        const auto batch_size = etl::dim<0>(v);
        return etl::force_temporary(etl::rep_l(etl::rep<NH1, NH2>(b), batch_size));
    }

    template<typename H, cpp_enable_if(etl::all_fast<H>::value)>
    auto get_batch_c_rep(H&& /*h*/) const {
        static constexpr auto batch_size = etl::decay_traits<H>::template dim<0>();
        return etl::force_temporary(etl::rep_l<batch_size>(etl::rep<NV1, NV2>(c)));
    }

    template<typename H, cpp_disable_if(etl::all_fast<H>::value)>
    auto get_batch_c_rep(H&& h) const {
        const auto batch_size = etl::dim<0>(h);
        return etl::force_temporary(etl::rep_l(etl::rep<NV1, NV2>(c), batch_size));
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

    template <typename V1, typename V2, size_t Off = 0, cpp_enable_if(etl::all_fast<V1, V2>::value)>
    static void validate_inputs() {
        static_assert(etl::decay_traits<V1>::dimensions() == 3 + Off, "Inputs must be 3D");
        static_assert(etl::decay_traits<V2>::dimensions() == 3 + Off, "Inputs must be 3D");

        static_assert(etl::decay_traits<V1>::template dim<0 + Off>() == NC, "Invalid number of input channels");
        static_assert(etl::decay_traits<V1>::template dim<1 + Off>() == NV1, "Invalid input dimensions");
        static_assert(etl::decay_traits<V1>::template dim<2 + Off>() == NV2, "Invalid input dimensions");

        static_assert(etl::decay_traits<V2>::template dim<0 + Off>() == NC, "Invalid number of input channels");
        static_assert(etl::decay_traits<V2>::template dim<1 + Off>() == NV1, "Invalid input dimensions");
        static_assert(etl::decay_traits<V2>::template dim<2 + Off>() == NV2, "Invalid input dimensions");
    }

    template <typename H1, typename H2, size_t Off = 0, cpp_enable_if(etl::all_fast<H1, H2>::value)>
    static void validate_outputs() {
        static_assert(etl::decay_traits<H1>::dimensions() == 3 + Off, "Outputs must be 3D");
        static_assert(etl::decay_traits<H2>::dimensions() == 3 + Off, "Outputs must be 3D");

        static_assert(etl::decay_traits<H1>::template dim<0 + Off>() == K, "Invalid number of output channels");
        static_assert(etl::decay_traits<H1>::template dim<1 + Off>() == NH1, "Invalid output dimensions");
        static_assert(etl::decay_traits<H1>::template dim<2 + Off>() == NH2, "Invalid output dimensions");

        static_assert(etl::decay_traits<H2>::template dim<0 + Off>() == K, "Invalid number of output channels");
        static_assert(etl::decay_traits<H2>::template dim<1 + Off>() == NH1, "Invalid output dimensions");
        static_assert(etl::decay_traits<H2>::template dim<2 + Off>() == NH2, "Invalid output dimensions");
    }

    template <typename V1, typename V2, size_t Off = 0, cpp_disable_if(etl::all_fast<V1, V2>::value)>
    static void validate_inputs() {
        static_assert(etl::decay_traits<V1>::dimensions() == 3 + Off, "Inputs must be 3D");
        static_assert(etl::decay_traits<V2>::dimensions() == 3 + Off, "Inputs must be 3D");
    }

    template <typename H1, typename H2, size_t Off = 0, cpp_disable_if(etl::all_fast<H1, H2>::value)>
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
struct rbm_base_traits<conv_rbm<Desc>> {
    using desc      = Desc;
    using weight    = typename desc::weight;

    using input_one_t         = etl::fast_dyn_matrix<weight, desc::NC, desc::NV1, desc::NV2>;
    using output_one_t        = etl::fast_dyn_matrix<weight, desc::K, desc::NV1 - desc::NW1 + 1, desc::NV2 - desc::NW2 + 1>;
    using hidden_output_one_t = output_one_t;
    using input_t             = std::vector<input_one_t>;
    using output_t            = std::vector<output_one_t>;
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const size_t conv_rbm<Desc>::NV1;

template <typename Desc>
const size_t conv_rbm<Desc>::NV2;

template <typename Desc>
const size_t conv_rbm<Desc>::NH1;

template <typename Desc>
const size_t conv_rbm<Desc>::NH2;

template <typename Desc>
const size_t conv_rbm<Desc>::NC;

template <typename Desc>
const size_t conv_rbm<Desc>::NW1;

template <typename Desc>
const size_t conv_rbm<Desc>::NW2;

template <typename Desc>
const size_t conv_rbm<Desc>::K;

// Declare the traits for the RBM

template<typename Desc>
struct layer_base_traits<conv_rbm<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false;  ///< Indicates if the layer is dense
    static constexpr bool is_conv       = true; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = true;  ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = false; ///< Indicates if the layer is a patches layer
    static constexpr bool is_augment    = false; ///< Indicates if the layer is an augment layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = Desc::hidden_unit != dll::unit_type::SOFTMAX; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

template<typename Desc>
struct rbm_layer_base_traits<conv_rbm<Desc>> {
    using param = typename Desc::parameters;

    static constexpr bool has_momentum       = param::template contains<momentum>();                            ///< Does the RBM has momentum
    static constexpr bool has_clip_gradients = param::template contains<clip_gradients>();                      ///< Does the RBM has gradient clipping
    static constexpr bool is_parallel_mode   = param::template contains<parallel_mode>();                       ///< Does the RBM is in parallel
    static constexpr bool is_serial          = param::template contains<serial>();                              ///< Does the RBM is in serial mode
    static constexpr bool is_verbose         = param::template contains<verbose>();                             ///< Does the RBM is verbose
    static constexpr bool has_shuffle        = param::template contains<shuffle>();                             ///< Does the RBM has shuffle
    static constexpr bool is_dbn_only        = param::template contains<dbn_only>();                            ///< Does the RBM is only used inside a DBN
    static constexpr bool has_init_weights   = param::template contains<init_weights>();                        ///< Does the RBM use weights initialization
    static constexpr bool has_free_energy    = param::template contains<free_energy>();                         ///< Does the RBM displays the free energy
    static constexpr auto sparsity_method    = get_value_l<sparsity<dll::sparsity_method::NONE>, param>::value; ///< The RBM's sparsity method
    static constexpr auto bias_mode          = get_value_l<bias<dll::bias_mode::NONE>, param>::value;           ///< The RBM's sparsity bias mode
    static constexpr auto decay              = get_value_l<weight_decay<dll::decay_type::NONE>, param>::value;  ///< The RMB's sparsity decay type
    static constexpr bool has_sparsity       = sparsity_method != dll::sparsity_method::NONE;                   ///< Does the RBM has sparsity
};

/*!
 * \brief Specialization of the sgd_context for conv_rbm
 */
template <typename DBN, typename Desc>
struct sgd_context<DBN, conv_rbm<Desc>> {
    using layer_t = conv_layer<Desc>;
    using weight  = typename layer_t::weight;

    static constexpr size_t NV1 = layer_t::NV1;
    static constexpr size_t NV2 = layer_t::NV2;
    static constexpr size_t NH1 = layer_t::NH1;
    static constexpr size_t NH2 = layer_t::NH2;
    static constexpr size_t NW1 = layer_t::NW1;
    static constexpr size_t NW2 = layer_t::NW2;
    static constexpr size_t NC  = layer_t::NC;
    static constexpr size_t K   = layer_t::K;

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, K, NC, NW1, NW2> w_grad;
    etl::fast_matrix<weight, K> b_grad;

    etl::fast_matrix<weight, K, NC, NW1, NW2> w_inc;
    etl::fast_matrix<weight, K> b_inc;

    etl::fast_matrix<weight, batch_size, NC, NV1, NV2> input;
    etl::fast_matrix<weight, batch_size, K, NH1, NH2> output;
    etl::fast_matrix<weight, batch_size, K, NH1, NH2> errors;

    sgd_context()
            : w_inc(0.0), b_inc(0.0), output(0.0), errors(0.0) {}
};

} //end of dll namespace
