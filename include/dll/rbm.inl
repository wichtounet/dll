//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_RBM_INL
#define DLL_RBM_INL

#include "cpp_utils/assert.hpp"             //Assertions
#include "cpp_utils/stop_watch.hpp"         //Performance counter

#include "etl/etl.hpp"

#include "standard_rbm.hpp"
#include "tmp.hpp"
#include "layer_traits.hpp"

namespace dll {

/*!
 * \brief Standard version of Restricted Boltzmann Machine
 *
 * This follows the definition of a RBM by Geoffrey Hinton.
 */
template<typename Desc>
struct rbm final : public standard_rbm<rbm<Desc>, Desc> {
    using desc = Desc;
    using weight = typename desc::weight;
    using this_type = rbm<desc>;
    using base_type = standard_rbm<this_type, desc>;

    static constexpr const std::size_t num_visible = desc::num_visible;
    static constexpr const std::size_t num_hidden = desc::num_hidden;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static constexpr bool dbn_only = layer_traits<this_type>::is_dbn_only();

    //Weights and biases
    etl::fast_matrix<weight, num_visible, num_hidden> w;    //!< Weights
    etl::fast_vector<weight, num_hidden> b;                 //!< Hidden biases
    etl::fast_vector<weight, num_visible> c;                //!< Visible biases

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
    rbm() : standard_rbm<rbm<Desc>, Desc>(), b(0.0), c(0.0) {
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
        return "RBM: " + std::to_string(num_visible) + "(" + to_string(visible_unit) + ") -> " + std::to_string(num_hidden) + "(" + to_string(hidden_unit) + ")";
    }

    void display() const {
        std::cout << to_short_string() << std::endl;
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V>
    void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s) const {
        static etl::fast_matrix<weight, 1, num_hidden> t;
        base_type::template std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, t);
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V, typename T>
    void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s, T&& t) const {
        base_type::template std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, std::forward<T>(t));
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V, typename B, typename W>
    static void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s, const B& b, const W& w){
        static etl::fast_matrix<weight, 1, num_hidden> t;
        base_type::template std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, t);
    }

    template<bool P = true, bool S = true, typename H, typename V>
    void activate_visible(const H& h_a, const H& h_s, V&& v_a, V&& v_s) const {
        static etl::fast_matrix<weight, num_visible, 1> t;
        base_type::template std_activate_visible<P, S>(h_a, h_s, std::forward<V>(v_a), std::forward<V>(v_s), c, w, t);
    }

    template<bool P = true, bool S = true, typename H, typename V, typename T>
    void activate_visible(const H& h_a, const H& h_s, V&& v_a, V&& v_s, T&& t) const {
        base_type::template std_activate_visible<P, S>(h_a, h_s, std::forward<V>(v_a), std::forward<V>(v_s), c, w, std::forward<T>(t));
    }

    template<typename Sample, typename Output>
    void activation_probabilities(const Sample& item_data, Output& result){
        etl::dyn_vector<weight> item(item_data);

        static etl::dyn_vector<weight> next_s(num_hidden);

        activate_hidden(result, next_s, item, item);
    }

    template<typename Sample>
    etl::dyn_vector<weight> activation_probabilities(const Sample& item_data){
        etl::dyn_vector<weight> result(output_size());

        activation_probabilities(item_data, result);

        return result;
    }
};

//Allow odr-use of the constexpr static members

template<typename Desc>
const std::size_t rbm<Desc>::num_visible;

template<typename Desc>
const std::size_t rbm<Desc>::num_hidden;

} //end of dll namespace

#endif
