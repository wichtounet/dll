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

#include "standard_rbm.hpp"
#include "layer_traits.hpp"

#include "util/converter.hpp" //converter

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

    static constexpr const unit_type visible_unit = desc::visible_unit; ///< The type of visible units
    static constexpr const unit_type hidden_unit  = desc::hidden_unit;  ///< The type of hidden units

    static constexpr bool dbn_only = layer_traits<this_type>::is_dbn_only();

    template <std::size_t B>
    using input_batch_t = etl::fast_dyn_matrix<weight, B, num_visible>;

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

    //No copying
    rbm(const rbm& rbm) = delete;
    rbm& operator=(const rbm& rbm) = delete;

    //No moving
    rbm(rbm&& rbm) = delete;
    rbm& operator=(rbm&& rbm) = delete;

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

    void display() const {
        std::cout << to_short_string() << std::endl;
    }

    void backup_weights() {
        unique_safe_get(bak_w) = w;
        unique_safe_get(bak_b) = b;
        unique_safe_get(bak_c) = c;
    }

    void restore_weights() {
        w = *bak_w;
        b = *bak_b;
        c = *bak_c;
    }

    // Make base class them participate in overload resolution
    using base_type::activate_hidden;

    template <bool P = true, bool S = true, typename H1, typename H2, typename V>
    void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s) const {
        etl::fast_dyn_matrix<weight, num_hidden> t;
        base_type::template std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, t);
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V, typename T>
    void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s, T&& t) const {
        base_type::template std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, std::forward<T>(t));
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V, typename B, typename W>
    static void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s, const B& b, const W& w) {
        etl::fast_dyn_matrix<weight, num_hidden> t;
        base_type::template std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, t);
    }

    template <bool P = true, bool S = true, typename H, typename V>
    void activate_visible(const H& h_a, const H& h_s, V&& v_a, V&& v_s) const {
        etl::fast_dyn_matrix<weight, num_visible> t;
        base_type::template std_activate_visible<P, S>(h_a, h_s, std::forward<V>(v_a), std::forward<V>(v_s), c, w, t);
    }

    template <bool P = true, bool S = true, typename H, typename V, typename T>
    void activate_visible(const H& h_a, const H& h_s, V&& v_a, V&& v_s, T&& t) const {
        base_type::template std_activate_visible<P, S>(h_a, h_s, std::forward<V>(v_a), std::forward<V>(v_s), c, w, std::forward<T>(t));
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V>
    void batch_activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s) const {
        base_type::template batch_std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w);
    }

    template <bool P = true, bool S = true, typename H, typename V>
    void batch_activate_visible(const H& h_a, const H& h_s, V&& v_a, V&& v_s) const {
        base_type::template batch_std_activate_visible<P, S>(h_a, h_s, std::forward<V>(v_a), std::forward<V>(v_s), c, w);
    }

    template <typename Sample, typename Output>
    void activation_probabilities(const Sample& item_data, Output& result) const {
        etl::fast_dyn_vector<weight, num_visible> item(item_data);
        etl::fast_dyn_matrix<weight, num_hidden> next_s;

        activate_hidden(result, next_s, item, item);
    }

    template <typename Sample>
    etl::dyn_vector<weight> activation_probabilities(const Sample& item_data) const {
        etl::dyn_vector<weight> result(output_size());

        activation_probabilities(item_data, result);

        return result;
    }

    // activate_hidden(Output, Input)

    template <typename H, typename V>
    void activate_hidden(H&& h_a, const input_one_t& v_a) const {
        etl::fast_dyn_matrix<weight, num_hidden> t;
        base_type::template std_activate_hidden<true, false>(std::forward<H>(h_a), std::forward<H>(h_a), v_a, v_a, b, w, t);
    }

    template <typename H, typename Input>
    void activate_hidden(H&& h_a, const Input& v_a) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(*this, v_a);
        activate_hidden(h_a, converted);
    }

    // batch_activate_hidden(Output, Input)

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() == 2)>
    void batch_activate_hidden(H&& h_a, const V& v_a) const {
        base_type::template batch_std_activate_hidden<true, false>(std::forward<H>(h_a), std::forward<H>(h_a), v_a, v_a, b, w);
    }

    template <typename H, typename V, cpp_enable_if(etl::decay_traits<V>::dimensions() != 2)>
    void batch_activate_hidden(H&& h_a, const V& v_a) const {
        batch_activate_hidden(h_a, etl::reshape<etl::decay_traits<H>::template dim<0>(), num_visible>(v_a));
    }

    template<typename DRBM>
    static void dyn_init(DRBM& dyn){
        dyn.init_layer(num_visible, num_hidden);
        dyn.batch_size  = layer_traits<this_type>::batch_size();
    }

    void prepare_input(input_one_t&) const {
        // Nothing to do
    }

    template <std::size_t B>
    auto prepare_input_batch(){
        return etl::fast_dyn_matrix<weight, B, num_visible>();
    }

    template <std::size_t B>
    auto prepare_output_batch(){
        return etl::fast_dyn_matrix<weight, B, num_hidden>();
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
