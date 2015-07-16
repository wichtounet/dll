//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DENSE_LAYER_INL
#define DLL_DENSE_LAYER_INL

#include "cpp_utils/assert.hpp"             //Assertions

#include "etl/etl.hpp"

#include "tmp.hpp"
#include "layer_traits.hpp"

namespace dll {

template<typename Desc>
struct dense_layer final {
    using desc = Desc;
    using weight = typename desc::weight;
    using this_type = dense_layer<desc>;

    static constexpr const std::size_t num_visible = desc::num_visible;
    static constexpr const std::size_t num_hidden = desc::num_hidden;

    static constexpr const bool dbn_only = layer_traits<this_type>::is_dbn_only();

    static constexpr const function activation_function = desc::activation_function;

    //Weights and biases
    etl::fast_matrix<weight, num_visible, num_hidden> w;    //!< Weights
    etl::fast_vector<weight, num_hidden> b;                 //!< Hidden biases

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
    dense_layer(){
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

    static std::string to_short_string(){
        return "Dense: " + std::to_string(num_visible) + " -> (" + "TODO:function" + ") -> " + std::to_string(num_hidden);
    }

    void display() const {
        std::cout << to_short_string() << std::endl;
    }

    template<typename H1, typename V>
    void activate_hidden(H1&& output, const V& v) const {
        switch(activation_function){
            case function::SIGMOID:
                output = etl::sigmoid(b + v * w);
                break;
            case function::TANH:
                output = etl::tanh(b + v * w);
                break;
        }
    }

    //Utilities to be used by DBNs

    using input_one_t = etl::fast_dyn_matrix<weight, num_visible>;
    using output_one_t = etl::fast_dyn_matrix<weight, num_hidden>;
    using input_t = std::vector<input_one_t>;
    using output_t = std::vector<output_one_t>;

    template<typename Sample>
    input_one_t convert_sample(const Sample& sample) const {
        return input_one_t{sample};
    }

    template<typename Input>
    output_one_t prepare_one_output(bool /*is_last*/ = false, std::size_t /*labels*/ = 0) const {
        return {};
    }

    void activate_one(const input_one_t& input, output_one_t& h_a) const {
        activate_hidden(h_a, input);
    }
};

//Allow odr-use of the constexpr static members

template<typename Desc>
const std::size_t dense_layer<Desc>::num_visible;

template<typename Desc>
const std::size_t dense_layer<Desc>::num_hidden;

} //end of dll namespace

#endif
