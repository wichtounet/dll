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
    std::vector<Visible> visibles;
    std::vector<Hidden> hiddens;
    Bias bias_unit = 1;

    Weight* weights;
    Weight* a;
    Weight* b;

    std::size_t num_hidden;
    std::size_t num_visible;

    double learning_rate;

    std::mt19937_64 rand_engine;

    rbm(std::size_t num_hidden, std::size_t num_visible) : num_hidden(num_hidden), num_visible(num_visible) {
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
        for(size_t v = 0; v < num_visible; ++v){
            visibles[v] = items[v];
        }

        //1. Update the hidden states

        for(size_t j = 0; j < num_hidden; ++j){
            //sum = Sum(i)(v_i * w_ij)
            auto sum = 0.0;
            for(size_t i = 0; i < num_visible; ++i){
                sum += visibles[i] * w(i, j);
            }

            auto activation = b[j] + sum;

            //Probability of turning one
            auto p = logistic_sigmoid(activation);
            if(p > generator()){
                hiddens[j] = 1;
            } else {
                hiddens[j] = 0;
            }
        }

        std::vector<double> neg_probabilities(num_visible, 0.0);;

        //2. Update the visible states

        for(size_t i = 0; i < num_visible; ++i){
            //sum = Sum(j)(h_j * w_ij)
            auto sum = 0.0;
            for(size_t j = 0; j < num_hidden; ++j){
                sum += hiddens[j] * w(i, j);
            }

            auto activation = a[i] + sum;

            //Probability of turning one
            auto p = logistic_sigmoid(activation);
            if(p > generator()){
                visibles[i] = 1;
            } else {
                visibles[i] = 0;
            }

            neg_probabilities[i] = p;
        }



        //Update the states of the hidden units
        //auto energy_vh = -mul_sum(visibles, bias_visibles) - mul_sum(hiddens, bias_hidden);

        //auto pos_hidden_activations = dot(data, weights);
        //auto pos_hidden_probabilities = logistic_sigmoid(pos_hidden_activations);




    }
};

} //end of dbn namespace

#endif