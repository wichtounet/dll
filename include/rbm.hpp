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

#include "assert.hpp"

namespace dbn {

double logistic_sigmoid(double x){
    return 1 / (1 + exp(-x));
}

/*!
 * \brief Restricted Boltzmann Machine
 */
template<typename Visible, typename Hidden, int Batch = 1, int BatchSize = 1, bool Momentum = false>
struct rbm {
    typedef double weight;

    std::size_t num_visible;
    std::size_t num_hidden;

    std::vector<Visible> visibles;
    std::vector<Hidden> hiddens;

    weight* weights;
    weight* bias_visible;
    weight* bias_hidden;

    //Weights for momentum
    weight* weights_inc;
    weight* bias_visible_inc;
    weight* bias_hidden_inc;

    //TODO Add a way to configure that
    double learning_rate = 0.01;
    double momentum = 0.5;

    rbm(std::size_t num_visible, std::size_t num_hidden) :
            num_visible(num_visible), num_hidden(num_hidden),
            visibles(num_visible), hiddens(num_hidden) {

        weights = new weight[num_visible * num_hidden];

        //Initialize the weights using a Gaussian distribution of mean 0 and
        //variance 0.0.1
        std::mt19937_64 rand_engine(::time(nullptr));
        std::normal_distribution<double> distribution(0.0, 0.01);
        auto generator = std::bind(distribution, rand_engine);

        for(size_t v = 0; v < num_visible; ++v){
            for(size_t h = 0; h < num_hidden; ++h){
                w(v, h) = generator();
            }
        }

        //Init all the bias weights to zero
        bias_visible = new weight[num_visible];
        bias_hidden = new weight[num_hidden];

        std::fill(bias_visible, bias_visible + num_visible, 0.0);
        std::fill(bias_hidden, bias_hidden + num_hidden, 0.0);

        if(Momentum){
            weights_inc = new weight[num_visible * num_hidden];
            bias_visible_inc = new weight[num_visible];
            bias_hidden_inc = new weight[num_hidden];

            std::fill(weights_inc, weights_inc + num_hidden * num_visible, 0.0);
            std::fill(bias_visible_inc, bias_visible_inc + num_visible, 0.0);
            std::fill(bias_hidden_inc, bias_hidden_inc + num_hidden, 0.0);
        }
    }

    ~rbm(){
        delete[] weights;
        delete[] bias_visible;
        delete[] bias_hidden;

        if(Momentum){
            delete[] weights_inc;
            delete[] bias_visible_inc;
            delete[] bias_hidden_inc;
        }
    }

    inline weight& w(std::size_t i, std::size_t j){
        dbn_assert(i < num_visible, "i Out of bounds");
        dbn_assert(j < num_hidden, "j Out of bounds");

        return weights[num_hidden * i + j];
    }

    inline const weight& w(std::size_t i, std::size_t j) const {
        dbn_assert(i < num_visible, "i Out of bounds");
        dbn_assert(j < num_hidden, "j Out of bounds");

        return weights[num_hidden * i + j];
    }

    inline Visible& v(std::size_t i){
        dbn_assert(i < num_visible, "i Out of bounds");

        return visibles[i];
    }

    inline const Visible& v(std::size_t i) const {
        dbn_assert(i < num_visible, "i Out of bounds");

        return visibles[i];
    }

    inline Hidden& h(std::size_t j){
        dbn_assert(j < num_hidden, "j Out of bounds");

        return hiddens[j];
    }

    inline const Hidden& h(std::size_t j) const {
        dbn_assert(j < num_hidden, "j Out of bounds");

        return hiddens[j];
    }

    inline weight& a(std::size_t i){
        dbn_assert(i < num_visible, "i Out of bounds");

        return bias_visible[i];
    }

    inline const weight& a(std::size_t i) const {
        dbn_assert(i < num_visible, "i Out of bounds");

        return bias_visible[i];
    }

    inline weight& b(std::size_t j){
        dbn_assert(j < num_hidden, "j Out of bounds");

        return bias_hidden[j];
    }

    inline const weight& b(std::size_t j) const {
        dbn_assert(j < num_hidden, "j Out of bounds");

        return bias_hidden[j];
    }

    template<typename TrainingItem>
    void train(const std::vector<std::vector<TrainingItem>>& training_data, std::size_t max_epochs){
        //Initialize the visible biases to log(pi/(1-pi))
        for(size_t i = 0; i < num_visible; ++i){
            auto c = 0;
            for(auto& item : training_data){
                if(item[i] == 1){
                    ++c;
                }
            }

            auto pi = static_cast<double>(c) / training_data.size();
            bias_visible[i] = log(pi / (1 - pi));
        }

        std::mt19937_64 rand_engine(::time(NULL));
        std::uniform_int_distribution<size_t> distribution(0, training_data.size() - BatchSize);
        auto generator = std::bind(distribution, rand_engine);

        for(size_t epoch= 0; epoch < max_epochs; ++epoch){
            for(size_t i = 0; i < Batch; ++i){
                auto error = cd_step(training_data, generator());

                //std::cout << "epoch " << epoch << ", batch" << i << ": Reconstruction error: " << error << std::endl;
                std::cout << error << std::endl;
            }

            if(epoch == 10){
                momentum = 0.9;
            }
        }
    }

    const std::vector<double>& bernoulli(const std::vector<double>& input, std::vector<double>& output) const {
        static std::mt19937_64 rand_engine(::time(NULL));
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);
        static auto generator = bind(distribution, rand_engine);

        for(size_t i = 0; i < input.size(); ++i){
            output[i] = generator() < input[i] ? 1.0 : 0.0;
        }

        return output;
    }

    void activate_hidden(std::vector<double>& hiddens, const std::vector<double>& visibles) const {
        std::fill(hiddens.begin(), hiddens.end(), 0.0);

        for(size_t j = 0; j < num_hidden; ++j){
            double s = 0.0;
            for(size_t i = 0; i < num_visible; ++i){
                s += w(i, j) * visibles[i];
            }

            auto activation = b(j) + s;
            hiddens[j] = logistic_sigmoid(activation);
        }
    }

    void activate_visible(const std::vector<double>& hiddens, std::vector<double>& visibles) const {
        std::fill(visibles.begin(), visibles.end(), 0.0);

        for(size_t i = 0; i < num_visible; ++i){
            double s = 0.0;
            for(size_t j = 0; j < num_hidden; ++j){
                s += w(i, j) * hiddens[j];
            }

            auto activation = a(i) + s;
            visibles[i] = logistic_sigmoid(activation);
        }
    }

    template<typename TrainingItem>
    double cd_step(const std::vector<std::vector<TrainingItem>>& data, size_t batch_start){
        dbn_assert(!data.empty(), "Cannot train on empty batch");
        dbn_assert(batch_start + BatchSize <= data.size(), "Out of bounds");
        dbn_assert(data[batch_start].size() == num_visible, "The size of the training sample must match visible units");

        //Temporary data
        std::vector<double> v1(num_visible, 0.0);;
        std::vector<double> h1(num_hidden, 0.0);;
        std::vector<double> v2(num_visible, 0.0);;
        std::vector<double> h2(num_hidden, 0.0);;
        std::vector<double> hs(num_hidden, 0.0);;

        //Deltas
        std::vector<double> ga(num_visible, 0.0);;
        std::vector<double> gb(num_hidden, 0.0);;
        std::vector<double> gw(num_visible * num_hidden, 0.0);;

        for(size_t t = 0; t < BatchSize; ++t){
            auto & items = data[batch_start + t];

            for(size_t i = 0; i < num_visible; ++i){
                v1[i] = items[i];
            }

            activate_hidden(h1, v1);
            activate_visible(bernoulli(h1, hs), v2);
            activate_hidden(h2, v2);

            for(size_t i = 0; i < num_visible; ++i){
                for(size_t j = 0; j < num_hidden; ++j){
                    gw[j * num_visible + i] += h1[j] * v1[i] - h2[j] * v2[i];
                }
            }

            for(size_t i = 0; i < num_visible; ++i){
                ga[i] += v1[i] - v2[i];
            }

            for(size_t j = 0; j < num_hidden; ++j){
                gb[j] += h1[j] - h2[j];
            }
        }

        auto n_samples = static_cast<double>(BatchSize);

        //gw / BatchSize
        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                gw[j * num_visible + i] /= n_samples;
            }
        }

        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                weights_inc[j * num_visible + i] = weights_inc[j * num_visible + i] * momentum + gw[j * num_visible + i] * learning_rate;
            }
        }

        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                w(i,j) += weights_inc[j * num_visible + i];
            }
        }

        //ga /= BatchSize
        for(size_t i = 0; i < num_visible; ++i){
            ga[i] /= n_samples;
        }

        for(size_t i = 0; i < num_visible; ++i){
            bias_visible_inc[i] = bias_visible_inc[i] * momentum + learning_rate * ga[i];
        }

        for(size_t i = 0; i < num_visible; ++i){
            a(i) += bias_visible_inc[i];
        }

        //gb /= BatchSize
        for(size_t j = 0; j < num_hidden; ++j){
            gb[j] /= n_samples;
        }

        for(size_t j = 0; j < num_hidden; ++j){
            bias_hidden_inc[j] /= bias_hidden_inc[j] * momentum + learning_rate * gb[j];
        }

        for(size_t j = 0; j < num_hidden; ++j){
            b(j) += bias_hidden_inc[j];
        }

        //Compute the reconstruction error

        for(size_t i = 0; i < num_visible; ++i){
            ga[i] *= (1.0 / n_samples);
        }

        double error = 0.0;
        for(size_t i = 0; i < num_visible; ++i){
            error += ga[i] * ga[i];
        }
        error /= num_visible;
        error = sqrt(error);

        return error;
    }

    template<typename TrainingItem>
    void run_visible(const std::vector<TrainingItem>& items){
        dbn_assert(items.size() == num_visible, "The size of the training sample must match visible units");

        static std::mt19937_64 rand_engine(::time(nullptr));
        static std::uniform_real_distribution<> distribution(0.0, 1.0);
        static auto generator = std::bind(distribution, rand_engine);

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

            auto activation = b(j) + sum;

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
            printf("%-8ld %d\n", i, v(i));
        }
    }

    void display_hidden_units() const {
        std::cout << "Hidden Value" << std::endl;

        for(size_t j = 0; j < num_hidden; ++j){
            printf("%-8ld %d\n", j, h(j));
        }
    }
};

} //end of dbn namespace

#endif