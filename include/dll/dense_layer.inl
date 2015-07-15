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

    static constexpr bool dbn_only = layer_traits<this_type>::is_dbn_only();

    //Weights and biases
    etl::fast_matrix<weight, num_visible, num_hidden> w;    //!< Weights
    etl::fast_vector<weight, num_hidden> b;                 //!< Hidden biases

    conditional_fast_matrix_t<!dbn_only, weight, num_visible> v1; //!< State of the visible units
    conditional_fast_matrix_t<!dbn_only, weight, num_hidden> h1_a; //!< Activation probabilities of hidden units after first CD-step

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
    rbm() : standard_rbm<rbm<Desc>, Desc>(), b(0.0) {
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        w = etl::normal_generator<weight>() * 0.1;
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
        return "Dense: " + std::to_string(num_visible) + "(" + to_string(visible_unit) + ") -> " + std::to_string(num_hidden) + "(" + to_string(hidden_unit) + ")";
    }

    void display() const {
        std::cout << to_short_string() << std::endl;
    }
};

//Allow odr-use of the constexpr static members

template<typename Desc>
const std::size_t dense_desc<Desc>::num_visible;

template<typename Desc>
const std::size_t dense_desc<Desc>::num_hidden;

} //end of dll namespace

#endif
