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

#include "normal_rbm.hpp"
#include "checks.hpp"

namespace dll {

/*!
 * \brief Standard version of Restricted Boltzmann Machine
 *
 * This follows the definition of a RBM by Geoffrey Hinton.
 */
template<typename Desc>
struct rbm : public normal_rbm<rbm<Desc>, Desc> {
    typedef float weight;

    using desc = Desc;

    static constexpr const std::size_t num_visible = desc::num_visible;
    static constexpr const std::size_t num_hidden = desc::num_hidden;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    //Weights and biases
    etl::fast_matrix<weight, num_visible, num_hidden> w;    //!< Weights
    etl::fast_vector<weight, num_hidden> b;                 //!< Hidden biases
    etl::fast_vector<weight, num_visible> c;                //!< Visible biases

    //Reconstruction data
    etl::fast_vector<weight, num_visible> v1; //!< State of the visible units

    etl::fast_vector<weight, num_hidden> h1_a; //!< Activation probabilities of hidden units after first CD-step
    etl::fast_vector<weight, num_hidden> h1_s; //!< Sampled value of hidden units after first CD-step

    etl::fast_vector<weight, num_visible> v2_a; //!< Activation probabilities of visible units after first CD-step
    etl::fast_vector<weight, num_visible> v2_s; //!< Sampled value of visible units after first CD-step

    etl::fast_vector<weight, num_hidden> h2_a; //!< Activation probabilities of hidden units after last CD-step
    etl::fast_vector<weight, num_hidden> h2_s; //!< Sampled value of hidden units after last CD-step

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
    rbm() : normal_rbm<rbm<Desc>, Desc>(), b(0.0), c(0.0) {
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        w = etl::normal_generator<weight>() * 0.1;
    }

    static constexpr std::size_t input_size(){
        return num_visible;
    }

    static constexpr std::size_t output_size(){
        return num_hidden;
    }

    void display() const {
        std::cout << "RBM: " << num_visible << " -> " << num_hidden << std::endl;
    }

    template<typename H1, typename H2, typename V>
    void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s) const {
        return activate_hidden(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w);
    }

    template<typename H1, typename H2, typename V, typename T>
    void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s, T&& t) const {
        return activate_hidden(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, std::forward<T>(t));
    }

    template<typename H1, typename H2, typename V, typename B, typename W>
    static void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s, const B& b, const W& w){
        static etl::fast_matrix<weight, 1, num_hidden> t;

        activate_hidden(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, t);
    }

    template<typename H1, typename H2, typename V, typename B, typename W, typename T>
    static void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V&, const B& b, const W& w, T&& t){
        using namespace etl;

        if(hidden_unit == unit_type::BINARY){
            h_a = sigmoid(b + auto_vmmul(v_a, w, t));
            h_s = bernoulli(h_a);
        } else if(hidden_unit == unit_type::EXP){
            h_a = exp(b + auto_vmmul(v_a, w, t));
            h_s = bernoulli(h_a);
        } else if(hidden_unit == unit_type::RELU){
            h_a = max(b + auto_vmmul(v_a, w, t), 0.0);
            h_s = logistic_noise(h_a);
        } else if(hidden_unit == unit_type::RELU6){
            h_a = min(max(b + auto_vmmul(v_a, w, t), 0.0), 6.0);
            h_s = ranged_noise(h_a, 6.0);
        } else if(hidden_unit == unit_type::RELU1){
            h_a = min(max(b + auto_vmmul(v_a, w, t), 0.0), 1.0);
            h_s = ranged_noise(h_a, 1.0);
        } else if(hidden_unit == unit_type::SOFTMAX){
            h_a = softmax(b + auto_vmmul(v_a, w, t));
            h_s = one_if_max(h_a);
        } else {
            cpp_unreachable("Invalid path");
        }

        //TODO nan_check_deep(h_a);
        //TODO nan_check_deep(h_s);
    }

    template<typename H, typename V>
    void activate_visible(const H& h_a, const H& h_s, V&& v_a, V&& v_s) const {
        using namespace etl;

        static fast_matrix<weight, num_visible, 1> t;

        activate_visible(h_a, h_s, std::forward<V>(v_a), std::forward<V>(v_s), t);
    }

    template<typename H, typename V, typename T>
    void activate_visible(const H&, const H& h_s, V&& v_a, V&& v_s, T&& t) const {
        using namespace etl;

        if(visible_unit == unit_type::BINARY){
            v_a = sigmoid(c + auto_vmmul(w, h_s, t));
            v_s = bernoulli(v_a);
        } else if(visible_unit == unit_type::GAUSSIAN){
            v_a = c + auto_vmmul(w, h_s, t);
            v_s = v_a;
        } else if(visible_unit == unit_type::RELU){
            v_a = max(c + auto_vmmul(w, h_s, t), 0.0);
            v_s = logistic_noise(v_a);
        } else {
            cpp_unreachable("Invalid path");
        }

        //TODO nan_check_deep(v_a);
        //TODO nan_check_deep(v_s);
    }
};

} //end of dbn namespace

#endif
