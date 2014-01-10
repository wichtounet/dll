//=======================================================================
// Copyright Baptiste Wicht 2014.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#ifndef DBN_RBM_HPP
#define DBN_RBM_HPP

#include <cmath>
#include <vector>
#include <random>
#include <functional>

namespace dbn {

template<typename A, typename B, typename Ret = double>
struct mul {
    constexpr Ret operator()(const A& a, const B& b) const {
        return a * b;
    }
};

template<typename A, typename B, typename Operator>
double sum(const std::vector<A>& as, B* bs, Operator op){
    double acc = 0;

    for(size_t i = 0; i < as.size(); ++i){
        acc += op(as[i], bs[i]);
    }

    return acc;
}

template<typename A, typename B>
double mul_sum(const std::vector<A>& as, B* bs){
    return sum(as, bs, mul<A, B>());
}

double logistic_sigmoid(double x){
    return 1 / (1 / exp(-x));
}

/*!
 * \brief Restricted Boltzmann Machine
 */
template<typename Visible, typename Hidden, typename Bias, typename Weight = double>
struct rbm {
    std::size_t num_visible;
    std::size_t num_hidden;

    std::vector<Visible> visibles;
    std::vector<Hidden> hiddens;
    Bias bias_unit = 1;

    Weight* weights;
    Weight* a;
    Weight* b;

    double learning_rate;

    std::mt19937_64 rand_engine;

    rbm(std::size_t num_visible, std::size_t num_hidden) :
            num_visible(num_visible), num_hidden(num_hidden),
            visibles(num_visible), hiddens(num_hidden) {

        weights = new Weight[num_visible * num_hidden];

        std::normal_distribution<double> distribution(0.0, 0.1);
        auto generator = std::bind(distribution, rand_engine);

        for(size_t v = 0; v < num_visible; ++v){
            for(size_t h = 0; h < num_hidden; ++h){
                w(v, h) = generator();
            }
        }

        //Init all the bias weights to zero
        a = new Weight[num_visible]();
        b = new Weight[num_hidden]();
    }

    inline Weight& w(std::size_t i, std::size_t j){
        return weights[num_visible * i + j];
    }

    inline Visible& v(std::size_t i){
        return visibles[i];
    }

    inline Hidden& h(std::size_t j){
        return hiddens[j];
    }

    inline const Visible& v(std::size_t i) const {
        return visibles[i];
    }

    inline const Hidden& h(std::size_t j) const {
        return hiddens[j];
    }

    ~rbm(){
        delete[] weights;
        delete[] a;
        delete[] b;
    }

    template<typename TrainingItem>
    void train(const std::vector<std::vector<TrainingItem>>& training_data, std::size_t max_epochs){
        std::uniform_int_distribution<> distribution(0, training_data.size() - 1);
        auto generator = std::bind(distribution, rand_engine);

        for(size_t i = 0; i < max_epochs; ++i){
            epoch(training_data[generator()]);
        }
    }

    template<typename TrainingItem>
    void epoch(const std::vector<TrainingItem>& items){
        std::uniform_real_distribution<> distribution(0.0, 1.0);
        auto generator = std::bind(distribution, rand_engine);

        //Size should match

        // ??????? Set the states of the visible units
        for(size_t i = 0; i < num_visible; ++i){
            v(i) = items[i];
        }

        std::vector<double> pos_hidden_p(num_hidden, 0.0);;
        std::vector<double> neg_visible_p(num_visible, 0.0);;
        std::vector<double> neg_hidden_p(num_hidden, 0.0);;

        //This is the "positive CD phase" (reality phase)

        //1. Update the hidden states from the visibles states

        for(size_t j = 0; j < num_hidden; ++j){
            //sum = Sum(i)(v_i * w_ij)
            auto sum = 0.0;
            for(size_t i = 0; i < num_visible; ++i){
                sum += v(i) * w(i, j);
            }

            auto activation = b[j] + sum;

            //Probability of turning one
            auto p = logistic_sigmoid(activation);
            if(p > generator()){
                h(j) = 1;
            } else {
                h(j) = 0;
            }

            pos_hidden_p[j] = p;
        }

        //This is the "negative CD phase" (daydream phase)

        //2. Reconstruct the visible units

        for(size_t i = 0; i < num_visible; ++i){
            //sum = Sum(j)(h_j * w_ij)
            auto sum = 0.0;
            for(size_t j = 0; j < num_hidden; ++j){
                //TODO Check if we really need hiddens[j] here
                sum += h(j) * w(i, j);
            }

            auto activation = a[i] + sum;

            //Probability of turning one
            auto p = logistic_sigmoid(activation);
            if(p > generator()){
                v(i) = 1;
            } else {
                v(i) = 0;
            }

            neg_visible_p[i] = p;
        }

        //3. Sample again from the hidden units

        //This time, the hidden units are sampled from the probabilities
        //and not the stochastic state of the visible units
        for(size_t j = 0; j < num_hidden; ++j){
            //sum = Sum(i)(v_i * w_ij)
            auto sum = 0.0;
            for(size_t i = 0; i < num_visible; ++i){
                sum += neg_visible_p[i] * w(i, j);
            }

            auto activation = b[j] + sum;

            //Probability of turning one
            auto p = logistic_sigmoid(activation);
            if(p > generator()){
                h(j) = 1;
            } else {
                h(j) = 0;
            }

            neg_hidden_p[j] = p;
        }

        //4. Update the weights by using the gradients

        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                w(i,j) += learning_rate * (pos_hidden_p[j] * items[i] - neg_hidden_p[j] * neg_visible_p[i]);
            }
        }

        //TODO Update bias weights
    }

    template<typename TrainingItem>
    void run_visible(const std::vector<TrainingItem>& items){
        std::uniform_real_distribution<> distribution(0.0, 1.0);
        auto generator = std::bind(distribution, rand_engine);

        //Set the state of the visible units
        for(size_t i = 0; i < num_visible; ++i){
            v(i) = items[i];
        }

        //Sample the hidden units from the visible units
        for(size_t j = 0; j < num_hidden; ++j){
            //sum = Sum(i)(v_i * w_ij)
            auto sum = 0.0;
            for(size_t i = 0; i < num_visible; ++i){
                sum += v(i) * w(i, j);
            }

            auto activation = b[j] + sum;

            //Probability of turning one
            auto p = logistic_sigmoid(activation);
            if(p > generator()){
                h(j) = 1;
            } else {
                h(j) = 0;
            }
        }
    }

    void display() const {
        display_visible_units();
        display_hidden_units();
    }

    void display_visible_units() const {
        std::cout << "Visible  Value" << std::endl;

        for(size_t i = 0; i < num_visible; ++i){
            printf("%ld %d\n", i, v(i));
        }
    }

    void display_hidden_units() const {
        std::cout << "Hidden Value" << std::endl;

        for(size_t j = 0; j < num_visible; ++j){
            printf("%ld %d\n", j, h(j));
        }
    }
};

} //end of dbn namespace

#endif