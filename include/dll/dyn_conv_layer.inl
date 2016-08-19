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
 * \brief Standard dynamic convolutional layer of neural network.
 */
template <typename Desc>
struct dyn_conv_layer final : neural_base<dyn_conv_layer<Desc>> {
    using desc      = Desc;                  ///< The descriptor type
    using weight    = typename desc::weight; ///< The weight type
    using this_type = dyn_conv_layer<desc>;  ///< This type

    static constexpr const bool dbn_only = layer_traits<this_type>::is_dbn_only();

    static constexpr const function activation_function = desc::activation_function;

    using input_one_t  = etl::dyn_matrix<weight, 3>; ///< The type for one input
    using output_one_t = etl::dyn_matrix<weight, 3>; ///< The type for one output
    using input_t      = std::vector<input_one_t>;   ///< The type for many input
    using output_t     = std::vector<output_one_t>;  ///< The type for many output

    //TODO CHECK
    template <std::size_t B>
    using input_batch_t = etl::fast_dyn_matrix<weight, B, 1>;

    using w_type = etl::dyn_matrix<weight, 4>;
    using b_type = etl::dyn_matrix<weight, 1>;

    //Weights and biases
    w_type w; //!< Weights
    b_type b; //!< Hidden biases

    //Backup weights and biases
    std::unique_ptr<w_type> bak_w; //!< Backup Weights
    std::unique_ptr<b_type> bak_b; //!< Backup Hidden biases

    //No copying
    dyn_conv_layer(const dyn_conv_layer& layer) = delete;
    dyn_conv_layer& operator=(const dyn_conv_layer& layer) = delete;

    //No moving
    dyn_conv_layer(dyn_conv_layer&& layer) = delete;
    dyn_conv_layer& operator=(dyn_conv_layer&& layer) = delete;

    size_t nv1; ///< The first visible dimension
    size_t nv2; ///< The second visible dimension
    size_t nh1; ///< The first output dimension
    size_t nh2; ///< The second output dimension
    size_t nc;  ///< The number of input channels
    size_t k;   ///< The number of filters

    size_t nw1; ///< The first dimension of the filters
    size_t nw2; ///< The second dimension of the filters

    size_t batch_size = 25;

    dyn_conv_layer(){
        // Nothing else to init
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

        w = etl::dyn_matrix<weight, 4>(nc, k, nw1, nw2);

        b = etl::dyn_vector<weight>(k);

        //Initialize the weights and biases following Lecun approach
        //to initialization [lecun-98b]

        w = etl::normal_generator<weight>() * std::sqrt(2.0 / double(nc * nv1 * nv2));

        if (activation_function == function::RELU) {
            b = 0.01;
        } else {
            b = etl::normal_generator<weight>() * std::sqrt(2.0 / double(nc * nv1 * nv2));
        }
    }

    std::size_t input_size() const noexcept {
        return nc * nv1 * nv2;
    }

    std::size_t output_size() const noexcept {
        return k * nh1 * nh2;
    }

    std::size_t parameters() const noexcept {
        return k * nw1 * nw2;
    }

    std::string to_short_string() const {
        char buffer[1024];
        snprintf(buffer, 1024, "Conv(dyn): %lux%lux%lu -> (%lux%lux%lu) -> %s -> %lux%lux%lu", nc, nv1, nv2, k, nw1, nw2, to_string(activation_function).c_str(), k, nh1, nh2);
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
        etl::dyn_matrix<weight, 4> v_cv(2UL, k, nh1, nh2); //Temporary convolution

        auto w_f = etl::force_temporary(w);

        //flip all the kernels horizontally and vertically

        for (std::size_t channel = 0; channel < nc; ++channel) {
            for (size_t k = 0; k < k; ++k) {
                w_f(channel)(k).fflip_inplace();
            }
        }

        v_cv(1) = 0;

        for (std::size_t channel = 0; channel < nc; ++channel) {
            etl::conv_2d_valid_multi(v(channel), w_f(channel), v_cv(0));

            v_cv(1) += v_cv(0);
        }

        output = f_activate<activation_function>(etl::rep(b, nh1, nh2) + v_cv(1));
    }

    template <typename H1, typename V>
    void batch_activate_hidden(H1&& output, const V& v) const {
        etl::dyn_matrix<weight, 4> v_cv(2UL, k, nh1, nh2); //Temporary convolution

        const auto Batch = etl::dim<0>(v);

        auto w_f = force_temporary(w);

        //flip all the kernels horizontally and vertically

        for (std::size_t channel = 0; channel < nc; ++channel) {
            for (size_t k = 0; k < k; ++k) {
                w_f(channel)(k).fflip_inplace();
            }
        }

        for (std::size_t batch = 0; batch < Batch; ++batch) {
            v_cv(1) = 0;

            for (std::size_t channel = 0; channel < nc; ++channel) {
                etl::conv_2d_valid_multi(v(batch)(channel), w_f(channel), v_cv(0));

                v_cv(1) += v_cv(0);
            }

            output(batch) = f_activate<activation_function>(etl::rep(b, nh1, nh2) + v_cv(1));
        }
    }

    void prepare_input(input_one_t& input) const {
        input = input_one_t(nc, nv1, nv2);
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
    auto prepare_input_batch(){
        return etl::dyn_matrix<weight, 4>(B, nc, nv1, nv2);
    }

    template <std::size_t B>
    auto prepare_output_batch(){
        return etl::dyn_matrix<weight, 4>(B, k, nh1, nh2);
    }

    template <typename DBN>
    void init_sgd_context() {
        this->sgd_context_ptr = std::make_shared<sgd_context<DBN, this_type>>(nc, nv1, nv2, k, nh1, nh2);
    }
};

} //end of dll namespace
