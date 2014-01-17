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
#include <fstream>

//TODO Find a better way to use mkdir
#include <sys/stat.h>

#include "assert.hpp"

namespace dbn {

template<typename T>
struct matrix {
    const size_t rows;
    const size_t columns;
    T* const _data;

    matrix(size_t rows, size_t columns) :
            rows(rows), columns(columns), _data(new T[rows * columns]){
        //Nothing else to init
    }

    matrix(size_t rows, size_t columns, const T& value) :
            rows(rows), columns(columns), _data(new T[rows * columns]){
        std::fill(_data, _data + size(), value);
    }

    matrix(const matrix& rhs) = delete;
    matrix& operator=(const matrix& rhs) = delete;

    ~matrix(){
        delete[] _data;
    }

    size_t size(){
        return rows * columns;
    }

    void operator=(const T& value){
        std::fill(_data, _data + size(), value);
    }

    T& operator()(size_t i, size_t j){
        dbn_assert(i < rows, "Out of bounds");
        dbn_assert(j < columns, "Out of bounds");

        return _data[i * columns + j];
    }

    const T& operator()(size_t i, size_t j) const {
        dbn_assert(i < rows, "Out of bounds");
        dbn_assert(j < columns, "Out of bounds");

        return _data[i * columns + j];
    }

    const T* data() const {
        return _data;
    }
};

double logistic_sigmoid(double x){
    return 1 / (1 + exp(-x));
}

/*!
 * \brief Restricted Boltzmann Machine
 */
template<typename Visible, typename Hidden, int BatchSize = 1, bool Momentum = false>
struct rbm {
    static_assert(BatchSize > 0, "Batch size must be at least 1");

    typedef double weight;

    std::size_t num_visible;
    std::size_t num_hidden;

    std::vector<Visible> visibles;
    std::vector<Hidden> hiddens;

    matrix<weight> w;
    weight* bias_visible;
    weight* bias_hidden;

    //Weights for momentum
    matrix<weight> w_inc;
    weight* bias_visible_inc;
    weight* bias_hidden_inc;

    //TODO Add a way to configure that
    double learning_rate = 0.01;
    double momentum = 0.2;

    rbm(std::size_t num_visible, std::size_t num_hidden) :
            num_visible(num_visible), num_hidden(num_hidden),
            visibles(num_visible), hiddens(num_hidden),
            w(num_visible, num_hidden), w_inc(num_visible, num_hidden) {

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
            w_inc = 0;

            bias_visible_inc = new weight[num_visible];
            bias_hidden_inc = new weight[num_hidden];

            std::fill(bias_visible_inc, bias_visible_inc + num_visible, 0.0);
            std::fill(bias_hidden_inc, bias_hidden_inc + num_hidden, 0.0);
        }
    }

    ~rbm(){
        delete[] bias_visible;
        delete[] bias_hidden;

        if(Momentum){
            delete[] bias_visible_inc;
            delete[] bias_hidden_inc;
        }
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

        auto batches = training_data.size() / BatchSize;

        for(size_t epoch= 0; epoch < max_epochs; ++epoch){
            double error = 0.0;
            for(size_t i = 0; i < batches; ++i){
                error += cd_step(training_data.begin() + i * BatchSize, training_data.begin() + (i+1) * BatchSize);
            }

            std::cout << "epoch " << epoch << ": Reconstruction error average: " << (error / batches) << std::endl;

            if(Momentum && epoch == 20){
                momentum = 0.7;
            }

            generate_hidden_images(epoch);
            generate_histograms(epoch);
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

    template<typename Iterator>
    double cd_step(Iterator it, Iterator end){
        dbn_assert(end - it == BatchSize, "Invalid size");
        dbn_assert(it->size() == num_visible, "The size of the training sample must match visible units");

        //Temporary data
        std::vector<double> v1(num_visible, 0.0);
        std::vector<double> h1(num_hidden, 0.0);
        std::vector<double> v2(num_visible, 0.0);
        std::vector<double> h2(num_hidden, 0.0);
        std::vector<double> hs(num_hidden, 0.0);

        //Deltas
        std::vector<double> ga(num_visible, 0.0);
        std::vector<double> gb(num_hidden, 0.0);
        matrix<double> gw(num_visible, num_hidden, 0.0);

        while(it != end){
            auto& items = *it++;

            for(size_t i = 0; i < num_visible; ++i){
                v1[i] = items[i] == 1 ? 1.0 : 0.0;
            }

            activate_hidden(h1, v1);
            activate_visible(bernoulli(h1, hs), v2);
            activate_hidden(h2, v2);

            for(size_t i = 0; i < num_visible; ++i){
                for(size_t j = 0; j < num_hidden; ++j){
                    gw(i, j) += h1[j] * v1[i] - h2[j] * v2[i];
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
                gw(i, j) /= n_samples;
            }
        }

        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                w_inc(i, j) = w_inc(i, j) * momentum + gw(i,j) * learning_rate;
            }
        }

        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                w(i,j) += w_inc(i, j);
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
            bias_hidden_inc[j] = bias_hidden_inc[j] * momentum + learning_rate * gb[j];
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

    void generate_hidden_images(size_t epoch){
        mkdir("reports", 0777);

        auto folder = "reports/epoch_" + std::to_string(epoch);
        mkdir(folder.c_str(), 0777);

        for(size_t j = 0; j < num_hidden; ++j){
            auto path = folder + "/h_" + std::to_string(j) + ".dat";
            std::ofstream file(path, std::ios::out);

            if(!file){
                std::cout << "Could not open file " << path << std::endl;
            }

            size_t i = num_visible;
            while(i > 0){
                --i;

                file << w(i, j) << " ";
            }

            file << std::endl;
            file.close();
        }
    }

    void generate_histograms(size_t epoch){
        mkdir("reports", 0777);

        auto folder = "reports/epoch_" + std::to_string(epoch);
        mkdir(folder.c_str(), 0777);

        generate_histogram(folder + "/weights.dat", w.data(), num_visible * num_hidden);
        generate_histogram(folder + "/visibles.dat", bias_visible, num_visible);
        generate_histogram(folder + "/hiddens.dat", bias_hidden, num_hidden);
    }

    void generate_histogram(const std::string& path, const double* weights, size_t size){
        std::ofstream file(path, std::ios::out);

        if(!file){
            std::cout << "Could not open file " << path << std::endl;
        }

        for(size_t i = 0; i < size; ++i){
            file << weights[i] << std::endl;
        }

        file << std::endl;
        file.close();
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