//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_RBM_INL
#define DBN_RBM_INL

#include <cmath>
#include <vector>
#include <random>
#include <functional>
#include <ctime>

#include "etl/fast_matrix.hpp"
#include "etl/fast_vector.hpp"
#include "etl/dyn_vector.hpp"
#include "etl/multiplication.hpp"

#include "rbm_base.hpp"      //The base class
#include "stop_watch.hpp"    //Performance counter
#include "assert.hpp"
#include "base_conf.hpp"
#include "math.hpp"
#include "io.hpp"

namespace dll {

template<typename RBM>
struct rbm_trainer;

/*!
 * \brief Standard version of Restricted Boltzmann Machine
 *
 * This follows the definition of a RBM by Geoffrey Hinton.
 */
template<typename Desc>
class rbm : public rbm_base<Desc> {
public:
    typedef double weight;
    typedef double value_t;

    using desc = Desc;

    static constexpr const std::size_t num_visible = desc::num_visible;
    static constexpr const std::size_t num_hidden = desc::num_hidden;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static constexpr const bool DBN = desc::DBN;

    static_assert(visible_unit != unit_type::SOFTMAX && visible_unit != unit_type::EXP,
        "Exponential and softmax Visible units are not support");
    static_assert(hidden_unit != unit_type::GAUSSIAN,
        "Gaussian hidden units are not supported");

    static constexpr const std::size_t num_visible_gra = DBN ? num_visible : 0;
    static constexpr const std::size_t num_hidden_gra = DBN ? num_hidden : 0;

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

    //Gradients computations for DBN
    etl::fast_matrix<weight, num_visible, num_hidden>& gr_w = w;
    etl::fast_vector<weight, num_hidden>& gr_b = b;

    etl::fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_incs;
    etl::fast_vector<weight, num_hidden_gra> gr_b_incs;

    etl::fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_best;
    etl::fast_vector<weight, num_hidden_gra> gr_b_best;

    etl::fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_best_incs;
    etl::fast_vector<weight, num_hidden_gra> gr_b_best_incs;

    etl::fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_df0;
    etl::fast_vector<weight, num_hidden_gra> gr_b_df0;

    etl::fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_df3;
    etl::fast_vector<weight, num_hidden_gra> gr_b_df3;

    etl::fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_s;
    etl::fast_vector<weight, num_hidden_gra> gr_b_s;

    etl::fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_tmp;
    etl::fast_vector<weight, num_hidden_gra> gr_b_tmp;

    std::vector<etl::dyn_vector<weight>> gr_probs_a;
    std::vector<etl::dyn_vector<weight>> gr_probs_s;

public:
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
    rbm() : b(0.0), c(0.0) {
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        static std::default_random_engine rand_engine(std::time(nullptr));
        static std::normal_distribution<weight> distribution(0.0, 1.0);
        static auto generator = std::bind(distribution, rand_engine);

        for(auto& weight : w){
            weight = generator() * 0.1;
        }

        //Better initialization of learning rate
        rbm_base<desc>::learning_rate =
                visible_unit == unit_type::GAUSSIAN && is_relu(hidden_unit) ? 1e-5
            :   visible_unit == unit_type::GAUSSIAN || is_relu(hidden_unit) ? 1e-3
            :   /* Only ReLU and Gaussian Units needs lower rate */           1e-1;
    }

    void store(std::ostream& os) const {
        binary_write_all(os, w);
        binary_write_all(os, b);
        binary_write_all(os, c);
    }

    void load(std::istream& is){
        binary_load_all(is, w);
        binary_load_all(is, b);
        binary_load_all(is, c);
    }

    template<typename Samples>
    double train(const Samples& training_data, std::size_t max_epochs){
        typedef typename std::remove_reference<decltype(*this)>::type this_type;

        dll::rbm_trainer<this_type> trainer;
        return trainer.train(*this, training_data, max_epochs);
    }

    template<typename Samples>
    void init_weights(const Samples& training_data){
        //Initialize the visible biases to log(pi/(1-pi))
        for(size_t i = 0; i < num_visible; ++i){
            auto count = std::count_if(training_data.begin(), training_data.end(),
                [i](auto& a){return a[i] == 1; });

            auto pi = static_cast<weight>(count) / training_data.size();
            pi += 0.0001;
            c(i) = log(pi / (1.0 - pi));

            dll_assert(std::isfinite(c(i)), "NaN verify");
        }
    }

    template<typename H, typename V>
    void activate_hidden(H& h_a, H& h_s, const V& v_a, const V& v_s) const {
        return activate_hidden(h_a, h_s, v_a, v_s, b, w);
    }

    template<bool Temp, typename H, typename V>
    void gr_activate_hidden(H& h_a, H& h_s, const V& v_a, const V& v_s) const {
        return activate_hidden(h_a, h_s, v_a, v_s, Temp ? gr_b_tmp : gr_b, Temp ? gr_w_tmp : gr_w);
    }

    template<typename H, typename V, typename B, typename W>
    static void activate_hidden(H& h_a, H& h_s, const V& v_a, const V&, const B& b, const W& w){
        static std::default_random_engine rand_engine(std::time(nullptr));

        using namespace etl;

        static fast_matrix<weight, 1, num_hidden> t;

        if(hidden_unit == unit_type::BINARY){
            h_a = sigmoid(b + mmul(reshape<1, num_visible>(v_a), w, t));
            h_s = bernoulli(h_a);
        } else if(hidden_unit == unit_type::EXP){
            h_a = exp(b + mmul(reshape<1, num_visible>(v_a), w, t));
            h_s = bernoulli(h_a);
        } else if(hidden_unit == unit_type::RELU){
            h_a = max(b + mmul(reshape<1, num_visible>(v_a), w, t), 0.0);
            h_s = logistic_noise(h_a);
        } else if(hidden_unit == unit_type::RELU6){
            h_a = min(max(b + mmul(reshape<1, num_visible>(v_a), w, t), 0.0), 6.0);
            h_s = ranged_noise(h_a, 6.0);
        } else if(hidden_unit == unit_type::RELU1){
            h_a = min(max(b + mmul(reshape<1, num_visible>(v_a), w, t), 0.0), 1.0);
            h_s = ranged_noise(h_a, 1.0);
        } else if(hidden_unit == unit_type::SOFTMAX){
            weight exp_sum = sum(exp(b + mmul(reshape<1, num_visible>(v_a), w, t)));

            h_a = exp(b + mmul(reshape<1, num_visible>(v_a), w, t)) / exp_sum;

            auto max = std::max_element(h_a.begin(), h_a.end());

            h_s = 0.0;
            h_s(std::distance(h_a.begin(), max)) = 1.0;
        } else {
            dll_unreachable("Invalid path");
        }

        nan_check_deep(h_a);
        nan_check_deep(h_s);
    }

    template<typename H, typename V>
    void activate_visible(const H&, const H& h_s, V& v_a, V& v_s) const {
        static std::default_random_engine rand_engine(std::time(nullptr));

        using namespace etl;

        static fast_matrix<weight, num_visible, 1> t;

        if(visible_unit == unit_type::BINARY){
            v_a = sigmoid(c + mmul(w, reshape<num_hidden, 1>(h_s), t));
            v_s = bernoulli(v_a);
        } else if(visible_unit == unit_type::GAUSSIAN){
            v_a = c + mmul(w, reshape<num_hidden, 1>(h_s), t);
            v_s = noise(v_a);
        } else if(visible_unit == unit_type::RELU){
            v_a = max(c + mmul(w, reshape<num_hidden, 1>(h_s), t), 0.0);
            v_s = noise(v_a);
        } else {
            dll_unreachable("Invalid path");
        }

        nan_check_deep(v_a);
        nan_check_deep(v_s);
    }

    weight free_energy() const {
        weight energy = 0.0;

        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                energy += w(i, j) * b(j) * c(i);
            }
        }

        return -energy;
    }

    template<typename Sample>
    void reconstruct(const Sample& items){
        dll_assert(items.size() == num_visible, "The size of the training sample must match visible units");

        stop_watch<> watch;

        //Set the state of the visible units
        v1 = items;

        activate_hidden(h1_a, h1_s, v1, v1);
        activate_visible(h1_a, h1_s, v2_a, v2_s);
        activate_hidden(h2_a, h2_s, v2_a, v2_s);

        std::cout << "Reconstruction took " << watch.elapsed() << "ms" << std::endl;
    }

    void display() const {
        display_visible_units();
        display_hidden_units();
    }

    void display_visible_units() const {
        std::cout << "Visible  Value" << std::endl;

        for(size_t i = 0; i < num_visible; ++i){
            printf("%-8lu %d\n", i, v2_s(i));
        }
    }

    void display_visible_units(size_t matrix) const {
        for(size_t i = 0; i < matrix; ++i){
            for(size_t j = 0; j < matrix; ++j){
                std::cout << v2_s(i * matrix + j) << " ";
            }
            std::cout << std::endl;
        }
    }

    void display_hidden_units() const {
        std::cout << "Hidden Value" << std::endl;

        for(size_t j = 0; j < num_hidden; ++j){
            printf("%-8lu %d\n", j, h2_s(j));
        }
    }

    void display_weights(){
        for(size_t j = 0; j < num_hidden; ++j){
            for(size_t i = 0; i < num_visible; ++i){
                std::cout << w(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    void display_weights(size_t matrix){
        for(size_t j = 0; j < num_hidden; ++j){
            for(size_t i = 0; i < num_visible;){
                for(size_t m = 0; m < matrix; ++m){
                    std::cout << w(i++, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
};

} //end of dbn namespace

#endif