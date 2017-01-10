//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/neural_layer.hpp"

namespace dll {

/*!
 * \brief Standard convolutional layer of neural network.
 */
template <typename Desc>
struct conv_layer final : neural_layer<conv_layer<Desc>, Desc> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = conv_layer<desc>;
    using base_type = neural_layer<this_type, desc>;

    static constexpr const std::size_t NV1 = desc::NV1; ///< The first dimension of the visible units
    static constexpr const std::size_t NV2 = desc::NV2; ///< The second dimension of the visible units
    static constexpr const std::size_t NH1 = desc::NH1; ///< The first dimension of the hidden units
    static constexpr const std::size_t NH2 = desc::NH2; ///< The second dimension of the hidden units
    static constexpr const std::size_t NC  = desc::NC;  ///< The number of input channels
    static constexpr const std::size_t K   = desc::K;   ///< The number of filters

    static constexpr const std::size_t NW1 = NV1 - NH1 + 1; //By definition
    static constexpr const std::size_t NW2 = NV2 - NH2 + 1; //By definition

    static constexpr const bool dbn_only = layer_traits<this_type>::is_dbn_only();

    static constexpr const function activation_function = desc::activation_function;

    using input_one_t  = etl::fast_dyn_matrix<weight, NC, NV1, NV2>;
    using output_one_t = etl::fast_dyn_matrix<weight, K, NH1, NH2>;
    using input_t      = std::vector<input_one_t>;
    using output_t     = std::vector<output_one_t>;

    using w_type = etl::fast_matrix<weight, K, NC, NW1, NW2>;
    using b_type = etl::fast_matrix<weight, K>;

    //Weights and biases
    w_type w; //!< Weights
    b_type b; //!< Hidden biases

    //Backup weights and biases
    std::unique_ptr<w_type> bak_w; //!< Backup Weights
    std::unique_ptr<b_type> bak_b; //!< Backup Hidden biases

    /*!
     * \brief Initialize a conv layer with basic weights.
     */
    conv_layer() : base_type() {
        //Initialize the weights and biases following Lecun approach
        //to initialization [lecun-98b]

        w = etl::normal_generator<weight>() * std::sqrt(2.0 / double(NC * NV1 * NV2));

        if (activation_function == function::RELU) {
            b = 0.01;
        } else {
            b = etl::normal_generator<weight>() * std::sqrt(2.0 / double(NC * NV1 * NV2));
        }
    }

    static constexpr std::size_t input_size() noexcept {
        return NC * NV1 * NV2;
    }

    static constexpr std::size_t output_size() noexcept {
        return K * NH1 * NH2;
    }

    static constexpr std::size_t parameters() noexcept {
        return K * NW1 * NW2;
    }

    static std::string to_short_string() {
        char buffer[1024];
        snprintf(buffer, 1024, "Conv: %lux%lux%lu -> (%lux%lux%lu) -> %s -> %lux%lux%lu", NC, NV1, NV2, K, NW1, NW2, to_string(activation_function).c_str(), K, NH1, NH2);
        return {buffer};
    }

    void activate_hidden(output_one_t& output, const input_one_t& v) const {
        auto b_rep = etl::force_temporary(etl::rep<NH1, NH2>(b));

        etl::reshape<1, K, NH1, NH2>(output) = etl::conv_4d_valid_flipped(etl::reshape<1, NC, NV1, NV2>(v), w);

        output = f_activate<activation_function>(b_rep + output);
    }

    template <typename V>
    void activate_hidden(output_one_t& output, const V& v) const {
        decltype(auto) converted = converter_one<V, input_one_t>::convert(*this, v);
        activate_hidden(output, converted);
    }

    template <typename H1, typename V>
    void batch_activate_hidden(H1&& output, const V& v) const {
        output = etl::conv_4d_valid_flipped(v, w);

        static constexpr const auto batch_size = etl::decay_traits<H1>::template dim<0>();

        auto b_rep = etl::force_temporary(etl::rep_l<batch_size>(etl::rep<NH1, NH2>(b)));

        output = f_activate<activation_function>(b_rep + output);
    }

    template <typename Input>
    output_one_t prepare_one_output() const {
        return {};
    }

    template <typename Input>
    static output_t prepare_output(std::size_t samples) {
        return output_t{samples};
    }

    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(NC, NV1, NV2, K, NH1, NH2);
    }

    template<typename C>
    void adapt_errors(C& context) const {
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
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const std::size_t conv_layer<Desc>::NV1;

template <typename Desc>
const std::size_t conv_layer<Desc>::NV2;

template <typename Desc>
const std::size_t conv_layer<Desc>::NH1;

template <typename Desc>
const std::size_t conv_layer<Desc>::NH2;

template <typename Desc>
const std::size_t conv_layer<Desc>::NC;

template <typename Desc>
const std::size_t conv_layer<Desc>::NW1;

template <typename Desc>
const std::size_t conv_layer<Desc>::NW2;

template <typename Desc>
const std::size_t conv_layer<Desc>::K;

} //end of dll namespace
