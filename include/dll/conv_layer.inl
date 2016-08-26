//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/assert.hpp" //Assertions

#include "etl/etl.hpp"

#include "neural_base.hpp"
#include "util/tmp.hpp"
#include "layer_traits.hpp"

namespace dll {

/*!
 * \brief Standard convolutional layer of neural network.
 */
template <typename Desc>
struct conv_layer final : neural_base<conv_layer<Desc>> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = conv_layer<desc>;

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

    template <std::size_t B>
    using input_batch_t = etl::fast_dyn_matrix<weight, B, NC, NV1, NV2>;

    using w_type = etl::fast_matrix<weight, NC, K, NW1, NW2>;
    using b_type = etl::fast_matrix<weight, K>;

    //Weights and biases
    w_type w; //!< Weights
    b_type b; //!< Hidden biases

    //Backup weights and biases
    std::unique_ptr<w_type> bak_w; //!< Backup Weights
    std::unique_ptr<b_type> bak_b; //!< Backup Hidden biases

    //No copying
    conv_layer(const conv_layer& layer) = delete;
    conv_layer& operator=(const conv_layer& layer) = delete;

    //No moving
    conv_layer(conv_layer&& layer) = delete;
    conv_layer& operator=(conv_layer&& layer) = delete;

    /*!
     * \brief Initialize a conv layer with basic weights.
     */
    conv_layer() {
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

    void display() const {
        std::cout << to_short_string() << std::endl;
    }

    void backup_weights() {
        unique_safe_get(bak_w) = w;
        unique_safe_get(bak_b) = b;
    }

    void restore_weights() {
        w = *bak_w;
        b = *bak_b;
    }

    template <typename V>
    void activate_hidden(output_one_t& output, const V& v) const {
        etl::fast_dyn_matrix<weight, 2, K, NH1, NH2> v_cv; //Temporary convolution

        auto w_f = etl::force_temporary(w);

        //flip all the kernels horizontally and vertically

        for (std::size_t channel = 0; channel < NC; ++channel) {
            for (size_t k = 0; k < K; ++k) {
                w_f(channel)(k).fflip_inplace();
            }
        }

        v_cv(1) = 0;

        for (std::size_t channel = 0; channel < NC; ++channel) {
            etl::conv_2d_valid_multi(v(channel), w_f(channel), v_cv(0));

            v_cv(1) += v_cv(0);
        }

        output = f_activate<activation_function>(etl::rep<NH1, NH2>(b) + v_cv(1));
    }

    template <typename H1, typename V>
    void batch_activate_hidden(H1&& output, const V& v) const {
        etl::fast_dyn_matrix<weight, 2, K, NH1, NH2> v_cv; //Temporary convolution

        const auto Batch = etl::dim<0>(v);

        auto w_f = force_temporary(w);

        //flip all the kernels horizontally and vertically

        for (std::size_t channel = 0; channel < NC; ++channel) {
            for (size_t k = 0; k < K; ++k) {
                w_f(channel)(k).fflip_inplace();
            }
        }

        for (std::size_t batch = 0; batch < Batch; ++batch) {
            v_cv(1) = 0;

            for (std::size_t channel = 0; channel < NC; ++channel) {
                etl::conv_2d_valid_multi(v(batch)(channel), w_f(channel), v_cv(0));

                v_cv(1) += v_cv(0);
            }

            output(batch) = f_activate<activation_function>(etl::rep<NH1, NH2>(b) + v_cv(1));
        }
    }

    template <typename Input>
    output_one_t prepare_one_output() const {
        return {};
    }

    template <typename Input>
    static output_t prepare_output(std::size_t samples) {
        return output_t{samples};
    }

    template <std::size_t B>
    auto prepare_input_batch(){
        return etl::fast_dyn_matrix<weight, B, NC, NV1, NV2>();
    }

    template <std::size_t B>
    auto prepare_output_batch(){
        return etl::fast_dyn_matrix<weight, B, K, NH1, NH2>();
    }

    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(NC, NV1, NV2, K, NH1, NH2);
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
