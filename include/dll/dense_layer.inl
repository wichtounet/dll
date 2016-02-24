//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DENSE_LAYER_INL
#define DLL_DENSE_LAYER_INL

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
struct dense_layer final : neural_base<dense_layer<Desc>> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = dense_layer<desc>;

    static constexpr const std::size_t num_visible = desc::num_visible;
    static constexpr const std::size_t num_hidden  = desc::num_hidden;

    static constexpr const bool dbn_only = layer_traits<this_type>::is_dbn_only();

    static constexpr const function activation_function = desc::activation_function;

    using input_one_t  = etl::fast_dyn_matrix<weight, num_visible>;
    using output_one_t = etl::fast_dyn_matrix<weight, num_hidden>;
    using input_t      = std::vector<input_one_t>;
    using output_t     = std::vector<output_one_t>;

    template <std::size_t B>
    using input_batch_t = etl::fast_dyn_matrix<weight, B, num_visible>;

    template <std::size_t B>
    using output_batch_t = etl::fast_dyn_matrix<weight, B, num_hidden>;

    using w_type = etl::fast_matrix<weight, num_visible, num_hidden>;
    using b_type = etl::fast_matrix<weight, num_hidden>;

    //Weights and biases
    w_type w; //!< Weights
    b_type b; //!< Hidden biases

    //Backup Weights and biases
    std::unique_ptr<w_type> bak_w; //!< Backup Weights
    std::unique_ptr<b_type> bak_b; //!< Backup Hidden biases

    //No copying
    dense_layer(const dense_layer& layer) = delete;
    dense_layer& operator=(const dense_layer& layer) = delete;

    //No moving
    dense_layer(dense_layer&& layer) = delete;
    dense_layer& operator=(dense_layer&& layer) = delete;

    /*!
     * \brief Initialize a dense layer with basic weights.
     *
     * The weights are initialized from a normal distribution of
     * zero-mean and unit variance.
     */
    dense_layer() {
        //Initialize the weights and biases following Lecun approach
        //to initialization [lecun-98b]

        b = etl::normal_generator<weight>(0.0, 1.0 / std::sqrt(double(num_visible)));
        w = etl::normal_generator<weight>(0.0, 1.0 / std::sqrt(double(num_visible)));
    }

    static constexpr std::size_t input_size() noexcept {
        return num_visible;
    }

    static constexpr std::size_t output_size() noexcept {
        return num_hidden;
    }

    static constexpr std::size_t parameters() noexcept {
        return num_visible * num_hidden;
    }

    static std::string to_short_string() {
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
        output = f_activate<activation_function>(b + etl::reshape<num_visible>(v) * w);
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
            auto expr = etl::force_temporary(etl::rep_l(b, Batch) + etl::reshape<Batch, num_visible>(input) * w);

            for (std::size_t i = 0; i < Batch; ++i) {
                output(i) = f_activate<activation_function>(expr(i));
            }
        } else {
            output = f_activate<activation_function>(etl::rep_l(b, Batch) + etl::reshape<Batch, num_visible>(input) * w);
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
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const std::size_t dense_layer<Desc>::num_visible;

template <typename Desc>
const std::size_t dense_layer<Desc>::num_hidden;

} //end of dll namespace

#endif
