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
 * \brief Standard dense layer of neural network.
 */
template <typename Desc>
struct dyn_dense_layer final : neural_base<dyn_dense_layer<Desc>> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = dyn_dense_layer<desc>;

    static constexpr const bool dbn_only = layer_traits<this_type>::is_dbn_only();

    static constexpr const function activation_function = desc::activation_function;

    using input_one_t  = etl::dyn_matrix<weight, 1>;
    using output_one_t = etl::dyn_matrix<weight, 1>;
    using input_t      = std::vector<input_one_t>;
    using output_t     = std::vector<output_one_t>;

    template <std::size_t B>
    using input_batch_t = etl::fast_dyn_matrix<weight, B, 1>; //TODO CHECK THAT

    template <std::size_t B>
    using output_batch_t = etl::fast_dyn_matrix<weight, B, 1>; //TODO CHECK THAT

    using w_type = etl::dyn_matrix<weight, 2>;
    using b_type = etl::dyn_matrix<weight, 1>;

    //Weights and biases
    w_type w; //!< Weights
    b_type b; //!< Hidden biases

    //Backup Weights and biases
    std::unique_ptr<w_type> bak_w; //!< Backup Weights
    std::unique_ptr<b_type> bak_b; //!< Backup Hidden biases

    //No copying
    dyn_dense_layer(const dyn_dense_layer& layer) = delete;
    dyn_dense_layer& operator=(const dyn_dense_layer& layer) = delete;

    //No moving
    dyn_dense_layer(dyn_dense_layer&& layer) = delete;
    dyn_dense_layer& operator=(dyn_dense_layer&& layer) = delete;

    std::size_t num_visible;
    std::size_t num_hidden;

    dyn_dense_layer() {}

    void init_layer(size_t nv, size_t nh) {
        num_visible = nv;
        num_hidden  = nh;

        w = etl::dyn_matrix<weight, 2>(num_visible, num_hidden);
        b = etl::dyn_matrix<weight, 1>(num_hidden);

        //Initialize the weights and biases following Lecun approach
        //to initialization [lecun-98b]

        w = etl::normal_generator<weight>(0.0, 1.0 / std::sqrt(double(num_visible)));
        b = etl::normal_generator<weight>(0.0, 1.0 / std::sqrt(double(num_visible)));
    }

    std::size_t input_size() const noexcept {
        return num_visible;
    }

    std::size_t output_size() const noexcept {
        return num_hidden;
    }

    std::size_t parameters() const noexcept {
        return num_visible * num_hidden;
    }

    std::string to_short_string() const {
        char buffer[1024];
        snprintf(buffer, 1024, "Dense: %lu -> %s -> %lu", num_visible, to_string(activation_function).c_str(), num_hidden);
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

    template <typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() == 1)>
    void activate_hidden(output_one_t& output, const V& v) const {
        output = f_activate<activation_function>(b + v * w);
    }

    template <typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() != 1)>
    void activate_hidden(output_one_t& output, const V& v) const {
        output = f_activate<activation_function>(b + etl::reshape(v, num_visible) * w);
    }

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() == 2)>
    void batch_activate_hidden(H&& output, const V& v) const {
        const auto Batch = etl::dim<0>(v);

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        if (activation_function == function::SOFTMAX) {
            auto expr = etl::force_temporary(etl::rep_l(b, Batch) + v * w);

            for (std::size_t i = 0; i < Batch; ++i) {
                output(i) = f_activate<activation_function>(expr(i));
            }
        } else {
            output = f_activate<activation_function>(etl::rep_l(b, Batch) + v * w);
        }
    }

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() != 2)>
    void batch_activate_hidden(H&& output, const V& input) const {
        constexpr const auto Batch = etl::decay_traits<V>::template dim<0>();

        cpp_assert(etl::dim<0>(output) == Batch, "The number of samples must be consistent");

        if (activation_function == function::SOFTMAX) {
            auto expr = etl::force_temporary(etl::rep_l(b, Batch) + etl::reshape(input, Batch, num_visible) * w);

            for (std::size_t i = 0; i < Batch; ++i) {
                output(i) = f_activate<activation_function>(expr(i));
            }
        } else {
            output = f_activate<activation_function>(etl::rep_l(b, Batch) + etl::reshape(input, Batch, num_visible) * w);
        }
    }

    template <typename DBN>
    void init_sgd_context() {
        this->sgd_context_ptr = std::make_shared<sgd_context<DBN, this_type>>(num_visible, num_hidden);
    }

    template <typename Input>
    output_one_t prepare_one_output() const {
        return output_one_t(num_hidden);
    }

    template <typename Input>
    output_t prepare_output(std::size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(std::size_t i = 0; i < samples; ++i){
            output.emplace_back(num_hidden);
        }
        return output;
    }

    template <std::size_t B>
    auto prepare_input_batch() const {
        return etl::dyn_matrix<weight>(B, num_visible);
    }

    template <std::size_t B>
    auto prepare_output_batch() const {
        return etl::dyn_matrix<weight>(B, num_hidden);
    }
};

} //end of dll namespace
