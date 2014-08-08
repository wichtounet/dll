//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DYN_RBM_INL
#define DLL_DYN_RBM_INL

#include <cmath>
#include <vector>
#include <random>
#include <functional>
#include <ctime>

#include "etl/dyn_matrix.hpp"
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
class dyn_rbm : public rbm_base<Desc> {
public:
    typedef double weight;
    typedef double value_t;

    using desc = Desc;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static_assert(visible_unit != unit_type::SOFTMAX && visible_unit != unit_type::EXP,
        "Exponential and softmax Visible units are not support");
    static_assert(hidden_unit != unit_type::GAUSSIAN,
        "Gaussian hidden units are not supported");

    //Weights and biases
    etl::dyn_matrix<weight> w;              //!< Weights
    etl::dyn_vector<weight> b;              //!< Hidden biases
    etl::dyn_vector<weight> c;              //!< Visible biases

    //Reconstruction data
    etl::dyn_vector<weight> v1; //!< State of the visible units

    etl::dyn_vector<weight> h1_a; //!< Activation probabilities of hidden units after first CD-step
    etl::dyn_vector<weight> h1_s; //!< Sampled value of hidden units after first CD-step

    etl::dyn_vector<weight> v2_a; //!< Activation probabilities of visible units after first CD-step
    etl::dyn_vector<weight> v2_s; //!< Sampled value of visible units after first CD-step

    etl::dyn_vector<weight> h2_a; //!< Activation probabilities of hidden units after last CD-step
    etl::dyn_vector<weight> h2_s; //!< Sampled value of hidden units after last CD-step

    const size_t num_visible;
    const size_t num_hidden;

    size_t batch_size = 25;

public:
    //No copying
    dyn_rbm(const dyn_rbm& rbm) = delete;
    dyn_rbm& operator=(const dyn_rbm& rbm) = delete;

    //No moving
    dyn_rbm(dyn_rbm&& rbm) = delete;
    dyn_rbm& operator=(dyn_rbm&& rbm) = delete;

    /*!
     * \brief Initialize a RBM with basic weights.
     *
     * The weights are initialized from a normal distribution of
     * zero-mean and 0.1 variance.
     */
    dyn_rbm(size_t num_visible, size_t num_hidden) : 
            w(num_visible, num_hidden), b(num_hidden, 0.0), c(num_visible, 0.0), 
            v1(num_visible), h1_a(num_hidden), h1_s(num_hidden), 
            v2_a(num_visible), v2_s(num_visible), h2_a(num_hidden), h2_s(num_hidden), 
            num_visible(num_visible), num_hidden(num_hidden) {
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

    template<typename H1, typename H2, typename V>
    void activate_hidden(H1& h_a, H2& h_s, const V& v_a, const V& v_s) const {
        return activate_hidden(h_a, h_s, v_a, v_s, b, w);
    }

    template<typename H1, typename H2, typename V, typename B, typename W>
    void activate_hidden(H1& h_a, H2& h_s, const V& v_a, const V&, const B& b, const W& w) const {
        static std::default_random_engine rand_engine(std::time(nullptr));

        using namespace etl;

        static dyn_matrix<weight> t(1, num_hidden);

        if(hidden_unit == unit_type::BINARY){
            h_a = sigmoid(b + mmul(reshape(v_a, 1, num_visible), w, t));
            h_s = bernoulli(h_a);
        } else if(hidden_unit == unit_type::EXP){
            h_a = exp(b + mmul(reshape(v_a, 1, num_visible), w, t));
            h_s = bernoulli(h_a);
        } else if(hidden_unit == unit_type::RELU){
            h_a = max(b + mmul(reshape(v_a, 1, num_visible), w, t), 0.0);
            h_s = logistic_noise(h_a);
        } else if(hidden_unit == unit_type::RELU6){
            h_a = min(max(b + mmul(reshape(v_a, 1, num_visible), w, t), 0.0), 6.0);
            h_s = ranged_noise(h_a, 6.0);
        } else if(hidden_unit == unit_type::RELU1){
            h_a = min(max(b + mmul(reshape(v_a, 1, num_visible), w, t), 0.0), 1.0);
            h_s = ranged_noise(h_a, 1.0);
        } else if(hidden_unit == unit_type::SOFTMAX){
            weight exp_sum = sum(exp(b + mmul(reshape(v_a, 1, num_visible), w, t)));

            h_a = exp(b + mmul(reshape(v_a, 1, num_visible), w, t)) / exp_sum;

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

        static dyn_matrix<weight> t(num_visible, 1);

        if(visible_unit == unit_type::BINARY){
            v_a = sigmoid(c + mmul(w, reshape(h_s, num_hidden, 1), t));
            v_s = bernoulli(v_a);
        } else if(visible_unit == unit_type::GAUSSIAN){
            v_a = c + mmul(w, reshape(h_s, num_hidden, 1), t);
            v_s = noise(v_a);
        } else if(visible_unit == unit_type::RELU){
            v_a = max(c + mmul(w, reshape(h_s, num_hidden, 1), t), 0.0);
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