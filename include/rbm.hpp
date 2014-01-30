//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_RBM_HPP
#define DBN_RBM_HPP

#include <cmath>
#include <vector>
#include <random>
#include <functional>
#include <fstream>

//TODO Find a better way to use mkdir
#include <sys/stat.h>

#include "assert.hpp"
#include "stop_watch.hpp"
#include "fast_matrix.hpp"
#include "fast_vector.hpp"

namespace dbn {

/*!
 * \brief Restricted Boltzmann Machine
 */
template<typename Layer, typename Conf>
class rbm {
public:
    typedef double weight;
    typedef double value_t;

    static constexpr const std::size_t num_visible = Layer::num_visible;
    static constexpr const std::size_t num_hidden = Layer::num_hidden;

    static constexpr const bool Momentum = Conf::Momentum;
    static constexpr const std::size_t BatchSize = Conf::BatchSize;
    static constexpr const bool Debug = Conf::Debug;

    static_assert(BatchSize > 0, "Batch size must be at least 1");

private:
    fast_vector<value_t, num_visible> visibles;
    fast_vector<value_t, num_hidden> hiddens;

    fast_matrix<weight, num_visible, num_hidden> w;
    fast_vector<weight, num_visible> a;
    fast_vector<weight, num_hidden> b;

    //Weights for momentum
    fast_matrix<weight, Momentum ? num_visible: 0, Momentum ? num_hidden : 0> w_inc;
    fast_vector<weight, Momentum ? num_visible : 0> a_inc;
    fast_vector<weight, Momentum ? num_hidden : 0> b_inc;

    //Temporary data
    fast_vector<weight, num_visible> v1;
    fast_vector<weight, num_hidden> h1;
    fast_vector<weight, num_visible> v2;
    fast_vector<weight, num_hidden> h2;
    fast_vector<weight, num_hidden> hs;

    //Deltas
    fast_matrix<weight, num_visible, num_hidden> gw;
    fast_vector<weight, num_visible> ga;
    fast_vector<weight, num_hidden> gb;

    //TODO Add a way to configure that
    double learning_rate = 0.1;
    double momentum = 0.5;

    void init_weights(){
        //Initialize the weights using a Gaussian distribution of mean 0 and
        //variance 0.0.1
        std::mt19937_64 rand_engine(::time(nullptr));
        std::normal_distribution<double> distribution(0.0, 1.0);
        auto generator = std::bind(distribution, rand_engine);

        for(size_t v = 0; v < num_visible; ++v){
            for(size_t h = 0; h < num_hidden; ++h){
                w(v, h) = generator() * 0.1;
            }
        }
    }

    static constexpr double logistic_sigmoid(double x){
        return 1.0 / (1.0 + exp(-x));
    }

    template<typename V1, typename V2>
    static const V2& bernoulli(const V1& input, V2& output){
        dbn_assert(input.size() == output.size(), "vector must the same sizes");

        static std::mt19937_64 rand_engine(::time(nullptr));
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);
        static auto generator = bind(distribution, rand_engine);

        for(size_t i = 0; i < input.size(); ++i){
            output(i) = generator() < input(i) ? 1.0 : 0.0;
        }

        return output;
    }

public:
    template<bool M = Momentum, typename std::enable_if<(!M), bool>::type = false>
    rbm() : a(0.0), b(0.0){
        static_assert(!Momentum, "This constructor should only be used without momentum support");

        init_weights();
    }

    template<bool M = Momentum, typename std::enable_if<(M), bool>::type = false>
    rbm() : a(0.0), b(0.0), w_inc(0.0){

        static_assert(Momentum, "This constructor should only be used with momentum support");

        init_weights();
    }

    template<typename TrainingItem>
    void train(const std::vector<std::vector<TrainingItem>>& training_data, std::size_t max_epochs){
        stop_watch<std::chrono::seconds> watch;

        //Initialize the visible biases to log(pi/(1-pi))
        for(size_t i = 0; i < num_visible; ++i){
            size_t c = 0;
            for(auto& items : training_data){
                if(items[i] == 1){
                    ++c;
                }
            }

            auto pi = static_cast<double>(c) / training_data.size();
            pi += 0.0001;
            a(i) = log(pi / (1.0 - pi));
        }

        auto batches = training_data.size() / BatchSize;

        for(size_t epoch= 0; epoch < max_epochs; ++epoch){
            double error = 0.0;
            for(size_t i = 0; i < batches; ++i){
                error += cd_step(training_data.begin() + i * BatchSize, training_data.begin() + (i+1) * BatchSize);
            }

            std::cout << "epoch " << epoch << ": Reconstruction error average: " << (error / batches) << " Free energy: " << free_energy() << std::endl;

            if(Momentum && epoch == 6){
                momentum = 0.9;
            }

            if(Debug){
                generate_hidden_images(epoch);
                generate_histograms(epoch);
            }
        }

        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;
    }

    template<typename V1, typename V2>
    void activate_hidden(V1& h, const V2& v) const {
        h = 0.0;

        for(size_t j = 0; j < num_hidden; ++j){
            double s = 0.0;
            for(size_t i = 0; i < num_visible; ++i){
                s += w(i, j) * v(i);
            }

            auto activation = b(j) + s;
            h(j) = logistic_sigmoid(activation);
        }
    }

    template<typename V1, typename V2>
    void activate_visible(const V1& h, V2& v) const {
        v = 0.0;

        for(size_t i = 0; i < num_visible; ++i){
            double s = 0.0;
            for(size_t j = 0; j < num_hidden; ++j){
                s += w(i, j) * h(j);
            }

            auto activation = a(i) + s;
            v(i) = logistic_sigmoid(activation);
        }
    }

    template<typename Iterator>
    double cd_step(Iterator it, Iterator end){
        dbn_assert(std::distance(it, end) == BatchSize, "Invalid size");
        dbn_assert(it->size() == num_visible, "The size of the training sample must match visible units");

        v1 = 0.0;
        h1 = 0.0;
        v2 = 0.0;
        h2 = 0.0;
        hs = 0.0;

        ga = 0.0;
        gb = 0.0;
        gw = 0.0;

        while(it != end){
            auto& items = *it++;

            for(size_t i = 0; i < num_visible; ++i){
                v1(i) = items[i];
            }

            activate_hidden(h1, v1);
            activate_visible(bernoulli(h1, hs), v2);
            activate_hidden(h2, v2);

            for(size_t i = 0; i < num_visible; ++i){
                for(size_t j = 0; j < num_hidden; ++j){
                    gw(i, j) += h1(j) * v1(i) - h2(j) * v2(i);
                }
            }

            ga += v1 - v2;
            gb += h1 - h2;
        }

        auto n_samples = static_cast<weight>(BatchSize);

        if(Momentum){
            w_inc = w_inc * momentum + (gw / n_samples) * learning_rate;
            w += w_inc;
        } else {
            w += (gw / n_samples) * learning_rate;
        }

        if(Momentum){
            a_inc = a_inc * momentum + (ga / n_samples) * learning_rate;
            a += a_inc;
        } else {
            a += (ga / n_samples) * learning_rate;
        }

        if(Momentum){
            b_inc = b_inc * momentum + (gb / n_samples) * learning_rate;
            b += b_inc;
        } else {
            b += (gb / n_samples) * learning_rate;
        }

        //Compute the reconstruction error

        double error = 0.0;
        for(size_t i = 0; i < num_visible; ++i){
            error += (ga(i) / n_samples) * (ga(i) / n_samples);
        }
        error = sqrt(error / num_visible);

        return error;
    }

    weight free_energy() const {
        weight energy = 0.0;

        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                energy += w(i, j) * b(j) * a(i);
            }
        }

        return -energy;
    }

    template<typename TrainingItem>
    void reconstruct(const std::vector<TrainingItem>& items){
        dbn_assert(items.size() == num_visible, "The size of the training sample must match visible units");

        stop_watch<> watch;

        //Set the state of the visible units
        for(size_t i = 0; i < num_visible; ++i){
            visibles(i) = items[i];
        }

        activate_hidden(h1, visibles);
        activate_visible(bernoulli(h1, hs), v1);
        bernoulli(v1, visibles);

        std::cout << "Reconstruction took " << watch.elapsed() << "ms" << std::endl;
    }

    void generate_hidden_images(size_t epoch){
        mkdir("reports", 0777);

        auto folder = "reports/epoch_" + std::to_string(epoch);
        mkdir(folder.c_str(), 0777);

        for(size_t j = 0; j < num_hidden; ++j){
            auto path = folder + "/h_" + std::to_string(j) + ".dat";
            std::ofstream file(path, std::ios::out);

            if(!file){
                std::cout << "Could not open file " << path << std::endl;
            } else {
                size_t i = num_visible;
                while(i > 0){
                    --i;

                    auto value = w(i,j);
                    file << static_cast<size_t>(value > 0 ? static_cast<size_t>(value * 255.0) << 8 : static_cast<size_t>(-value * 255.0) << 16) << " ";
                }

                file << std::endl;
                file.close();
            }
        }
    }

    void generate_histograms(size_t epoch){
        mkdir("reports", 0777);

        auto folder = "reports/epoch_" + std::to_string(epoch);
        mkdir(folder.c_str(), 0777);

        //generate_histogram(folder + "/weights.dat", w.data(), num_visible * num_hidden);
//        generate_histogram(folder + "/visibles.dat", a.data(), num_visible);
//        generate_histogram(folder + "/hiddens.dat", b.data(), num_hidden);

        if(Momentum){
            //generate_histogram(folder + "/weights_inc.dat", w_inc.data(), num_visible * num_hidden);
//            generate_histogram(folder + "/visibles_inc.dat", a_inc.data(), num_visible);
//            generate_histogram(folder + "/hiddens_inc.dat", b_inc.data(), num_hidden);
        }
    }

    void generate_histogram(const std::string& path, const double* weights, size_t size){
        std::ofstream file(path, std::ios::out);

        if(!file){
            std::cout << "Could not open file " << path << std::endl;
        } else {
            for(size_t i = 0; i < size; ++i){
                file << weights[i] << std::endl;
            }

            file << std::endl;
            file.close();
        }
    }

    void display() const {
        display_visible_units();
        display_hidden_units();
    }

    void display_visible_units() const {
        std::cout << "Visible  Value" << std::endl;

        for(size_t i = 0; i < num_visible; ++i){
            printf("%-8lu %d\n", i, visibles(i));
        }
    }

    void display_visible_units(size_t matrix) const {
        for(size_t i = 0; i < matrix; ++i){
            for(size_t j = 0; j < matrix; ++j){
                std::cout << visibles(i * matrix + j) << " ";
            }
            std::cout << std::endl;
        }
    }

    void display_hidden_units() const {
        std::cout << "Hidden Value" << std::endl;

        for(size_t j = 0; j < num_hidden; ++j){
            printf("%-8lu %d\n", j, hiddens(j));
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