//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementation of a Restricted Boltzmann Machine
 */

#pragma once

#include "cpp_utils/assert.hpp"     //Assertions
#include "cpp_utils/stop_watch.hpp" //Performance counter

#include "etl/etl.hpp"

#include "dll/rbm/standard_rbm.hpp"
#include "dll/layer_traits.hpp"

#include "dll/util/converter.hpp" //converter

namespace dll {

/*!
 * \brief Standard version of Restricted Boltzmann Machine
 *
 * This follows the definition of a RBM by Geoffrey Hinton.
 */
template <typename Desc>
struct rbm final : public standard_rbm<rbm<Desc>, Desc> {
    using desc      = Desc;                          ///< The layer descriptor
    using weight    = typename desc::weight;         ///< The weight type
    using this_type = rbm<desc>;                     ///< The type of this layer
    using base_type = standard_rbm<this_type, desc>; ///< The base type

    using input_t      = typename rbm_base_traits<this_type>::input_t;
    using output_t     = typename rbm_base_traits<this_type>::output_t;
    using input_one_t  = typename rbm_base_traits<this_type>::input_one_t;
    using output_one_t = typename rbm_base_traits<this_type>::output_one_t;

    static constexpr const std::size_t num_visible = desc::num_visible; ///< The number of visible units
    static constexpr const std::size_t num_hidden  = desc::num_hidden;  ///< The number of hidden units
    static constexpr const std::size_t batch_size  = desc::BatchSize;  ///< The mini-batch size

    static constexpr const unit_type visible_unit = desc::visible_unit; ///< The type of visible units
    static constexpr const unit_type hidden_unit  = desc::hidden_unit;  ///< The type of hidden units

    static constexpr bool dbn_only = layer_traits<this_type>::is_dbn_only();

    using w_type = etl::fast_matrix<weight, num_visible, num_hidden>; ///< The type used to store weights
    using b_type = etl::fast_vector<weight, num_hidden>;              ///< The type used to store hidden biases
    using c_type = etl::fast_vector<weight, num_visible>;             ///< The type used to store visible biases

    //Weights and biases
    w_type w; //!< Weights
    b_type b; //!< Hidden biases
    c_type c; //!< Visible biases

    //Backup weights and biases
    std::unique_ptr<w_type> bak_w; //!< Backup Weights
    std::unique_ptr<b_type> bak_b; //!< Backup Hidden biases
    std::unique_ptr<c_type> bak_c; //!< Backup Visible biases

    //Reconstruction data
    conditional_fast_matrix_t<!dbn_only, weight, num_visible> v1; //!< State of the visible units

    conditional_fast_matrix_t<!dbn_only, weight, num_hidden> h1_a; //!< Activation probabilities of hidden units after first CD-step
    conditional_fast_matrix_t<!dbn_only, weight, num_hidden> h1_s; //!< Sampled value of hidden units after first CD-step

    conditional_fast_matrix_t<!dbn_only, weight, num_visible> v2_a; //!< Activation probabilities of visible units after first CD-step
    conditional_fast_matrix_t<!dbn_only, weight, num_visible> v2_s; //!< Sampled value of visible units after first CD-step

    conditional_fast_matrix_t<!dbn_only, weight, num_hidden> h2_a; //!< Activation probabilities of hidden units after last CD-step
    conditional_fast_matrix_t<!dbn_only, weight, num_hidden> h2_s; //!< Sampled value of hidden units after last CD-step

    /*!
     * \brief Initialize a RBM with basic weights.
     *
     * The weights are initialized from a normal distribution of
     * zero-mean and 0.1 variance.
     */
    rbm()
            : standard_rbm<rbm<Desc>, Desc>(), b(0.0), c(0.0) {
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

    static std::string to_short_string() {
        return "RBM: " + std::to_string(num_visible) + "(" + to_string(visible_unit) + ") -> " + std::to_string(num_hidden) + "(" + to_string(hidden_unit) + ")";
    }

    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(num_visible, num_hidden);
        dyn.batch_size  = batch_size;
    }

    void prepare_input(input_one_t& input) const {
        // Need to initialize the dimensions of the matrix
        input = input_one_t(num_visible);
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
        // The reshape has no overhead, so better than SFINAE for nothing
        constexpr const auto Batch = etl::decay_traits<H>::template dim<0>();
        etl::reshape<Batch, num_visible>(output) = context.errors * etl::transpose(w);
    }

    template<typename C>
    void compute_gradients(C& context) const {
        context.w_grad = batch_outer(context.input, context.errors);
        context.b_grad = etl::sum_l(context.errors);
    }
};

/*!
 * \brief Simple traits to pass information around from the real
 * class to the CRTP class.
 */
template <typename Desc>
struct rbm_base_traits<rbm<Desc>> {
    using desc      = Desc;
    using weight    = typename desc::weight;

    using input_one_t  = etl::dyn_vector<weight>;
    using output_one_t = etl::dyn_vector<weight>;
    using input_t      = std::vector<input_one_t>;
    using output_t     = std::vector<output_one_t>;
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const std::size_t rbm<Desc>::num_visible;

template <typename Desc>
const std::size_t rbm<Desc>::num_hidden;

} //end of dll namespace
