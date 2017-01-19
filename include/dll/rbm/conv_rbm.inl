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

#include "standard_crbm.hpp" //The base class

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

    static constexpr const unit_type visible_unit = desc::visible_unit; ///< The type of visible unit
    static constexpr const unit_type hidden_unit  = desc::hidden_unit;  ///< The type of hidden unit

    static constexpr const std::size_t NV1 = desc::NV1; ///< The first dimension of the visible units
    static constexpr const std::size_t NV2 = desc::NV2; ///< The second dimension of the visible units
    static constexpr const std::size_t NH1 = desc::NH1; ///< The first dimension of the hidden units
    static constexpr const std::size_t NH2 = desc::NH2; ///< The second dimension of the hidden units
    static constexpr const std::size_t NC  = desc::NC;  ///< The number of input channels
    static constexpr const std::size_t K   = desc::K;   ///< The number of filters

    static constexpr const std::size_t batch_size  = desc::BatchSize;  ///< The mini-batch size

    static constexpr const std::size_t NW1 = NV1 - NH1 + 1; //By definition
    static constexpr const std::size_t NW2 = NV2 - NH2 + 1; //By definition

    static constexpr const bool dbn_only = layer_traits<this_type>::is_dbn_only();

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
    static constexpr std::size_t input_size() noexcept {
        return NV1 * NV2 * NC;
    }

    /*!
     * \brief Return the output size of the layer
     */
    static constexpr std::size_t output_size() noexcept {
        return NH1 * NH2 * K;
    }

    /*!
     * \brief Return the number of parameters of the layer
     */
    static constexpr std::size_t parameters() noexcept {
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
    static output_t prepare_output(std::size_t samples) {
        return output_t{samples};
    }

    template <typename Input>
    static output_one_t prepare_one_output() {
        return output_one_t{};
    }

    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(NC, NV1, NV2, K, NH1, NH2);
        dyn.batch_size  = batch_size;
    }

    template<typename C>
    void adapt_errors(C& context) const {
        static_assert(
            hidden_unit == unit_type::BINARY || hidden_unit == unit_type::RELU || hidden_unit == unit_type::SOFTMAX,
            "Only (C)RBM with binary, softmax or RELU hidden unit are supported");

        static constexpr const function activation_function =
            hidden_unit == unit_type::BINARY
                ? function::SIGMOID
                : (hidden_unit == unit_type::SOFTMAX ? function::SOFTMAX : function::RELU);

        context.errors = f_derivative<activation_function>(context.output) >> context.errors;
    }

    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        output = etl::conv_4d_full_flipped(context.errors, w);
    }

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
        static constexpr const auto batch_size = etl::decay_traits<V>::template dim<0>();
        return etl::force_temporary(etl::rep_l<batch_size>(etl::rep<NH1, NH2>(b)));
    }

    template<typename V, cpp_disable_if(etl::all_fast<V>::value)>
    auto get_batch_b_rep(V&& v) const {
        const auto batch_size = etl::dim<0>(v);
        return etl::force_temporary(etl::rep_l(etl::rep<NH1, NH2>(b), batch_size));
    }

    template<typename H, cpp_enable_if(etl::all_fast<H>::value)>
    auto get_batch_c_rep(H&& /*h*/) const {
        static constexpr const auto batch_size = etl::decay_traits<H>::template dim<0>();
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

    template <typename V1, typename V2, std::size_t Off = 0, cpp_enable_if(etl::all_fast<V1, V2>::value)>
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

    template <typename H1, typename H2, std::size_t Off = 0, cpp_enable_if(etl::all_fast<H1, H2>::value)>
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

    template <typename V1, typename V2, std::size_t Off = 0, cpp_disable_if(etl::all_fast<V1, V2>::value)>
    static void validate_inputs() {
        static_assert(etl::decay_traits<V1>::dimensions() == 3 + Off, "Inputs must be 3D");
        static_assert(etl::decay_traits<V2>::dimensions() == 3 + Off, "Inputs must be 3D");
    }

    template <typename H1, typename H2, std::size_t Off = 0, cpp_disable_if(etl::all_fast<H1, H2>::value)>
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
    using output_one_t        = etl::fast_dyn_matrix<weight, desc::K, desc::NH1, desc::NH2>;
    using hidden_output_one_t = output_one_t;
    using input_t             = std::vector<input_one_t>;
    using output_t            = std::vector<output_one_t>;
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const std::size_t conv_rbm<Desc>::NV1;

template <typename Desc>
const std::size_t conv_rbm<Desc>::NV2;

template <typename Desc>
const std::size_t conv_rbm<Desc>::NH1;

template <typename Desc>
const std::size_t conv_rbm<Desc>::NH2;

template <typename Desc>
const std::size_t conv_rbm<Desc>::NC;

template <typename Desc>
const std::size_t conv_rbm<Desc>::NW1;

template <typename Desc>
const std::size_t conv_rbm<Desc>::NW2;

template <typename Desc>
const std::size_t conv_rbm<Desc>::K;

} //end of dll namespace
