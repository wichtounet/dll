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
#include "stop_watch.hpp"

namespace dbn {

template<typename T>
class matrix {
private:
    const size_t rows;
    const size_t columns;
    T* const _data;

public:
    matrix() : rows(0), columns(0), _data(nullptr) {
        //Nothing else to init
    }

    matrix(size_t r, size_t c) :
            rows(r), columns(c), _data(new T[r* c]){
        //Nothing else to init
    }

    matrix(size_t r, size_t c, const T& value) :
            rows(r), columns(c), _data(new T[r * c]){
        std::fill(_data, _data + size(), value);
    }

    matrix(const matrix& rhs) = delete;
    matrix& operator=(const matrix& rhs) = delete;

    ~matrix(){
        delete[] _data;
    }

    size_t size() const {
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

template<typename T>
class vector {
private:
    const size_t rows;
    T* const _data;

public:
    vector() : rows(0), _data(nullptr){
        //Nothing else to init
    }

    vector(size_t r) : rows(r), _data(new T[r]){
        //Nothing else to init
    }

    vector(size_t r, const T& value) : rows(r), _data(new T[r]){
        std::fill(_data, _data + size(), value);
    }

    vector(const vector& rhs) = delete;
    vector& operator=(const vector& rhs) = delete;

    ~vector(){
        delete[] _data;
    }

    size_t size() const {
        return rows;
    }

    void operator=(const T& value){
        std::fill(_data, _data + size(), value);
    }

    T& operator()(size_t i){
        dbn_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    const T& operator()(size_t i) const {
        dbn_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    const T* data() const {
        return _data;
    }
};

/*!
 * \brief Restricted Boltzmann Machine
 */
template<bool Momentum = true, int BatchSize = 1>
class rbm {
public:
    static_assert(BatchSize > 0, "Batch size must be at least 1");

    typedef double weight;
    typedef double value_t;

    const std::size_t num_visible;
    const std::size_t num_hidden;

private:
    vector<value_t> visibles;
    vector<value_t> hiddens;

    matrix<weight> w;
    vector<weight> a;
    vector<weight> b;

    //Weights for momentum
    matrix<weight> w_inc;
    vector<weight> a_inc;
    vector<weight> b_inc;

    //Temporary data
    vector<weight> v1;
    vector<weight> h1;
    vector<weight> v2;
    vector<weight> h2;
    vector<weight> hs;

    //Deltas
    vector<weight> ga;
    vector<weight> gb;
    matrix<weight> gw;

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

    static double logistic_sigmoid(double x){
        return 1.0 / (1.0 + exp(-x));
    }

    static const vector<double>& bernoulli(const vector<double>& input, vector<double>& output){
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
    rbm(std::size_t nv, std::size_t nh) :
            num_visible(nv), num_hidden(nh),
            visibles(nv), hiddens(nh),
            w(nv, nh), a(nv, 0.0), b(nh, 0.0),
            v1(nv), h1(nh), v2(nv), h2(nh), hs(nh),
            ga(nv), gb(nh), gw(nv, nh) {

        static_assert(!Momentum, "This constructor should only be used without momentum support");

        init_weights();
    }

    template<bool M = Momentum, typename std::enable_if<(M), bool>::type = false>
    rbm(std::size_t nv, std::size_t nh) :
            num_visible(nv), num_hidden(nh),
            visibles(nv), hiddens(nh),
            w(nv, nh), a(nv, 0.0), b(nh, 0.0),
            w_inc(nv, nh, 0.0), a_inc(nv, 0.0), b_inc(nh, 0.0),
            v1(nv), h1(nh), v2(nv), h2(nh), hs(nh),
            ga(nv), gb(nh), gw(nv, nh) {

        static_assert(Momentum, "This constructor should only be used with momentum support");

        init_weights();
    }

    template<typename TrainingItem>
    void train(const std::vector<std::vector<TrainingItem>>& training_data, std::size_t max_epochs){
        stop_watch<std::chrono::seconds> watch;

        //Initialize the visible biases to log(pi/(1-pi))
        for(size_t i = 0; i < num_visible; ++i){
            size_t c = 0;
            for(auto& item : training_data){
                if(item[i] == 1){
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

            std::cout << "epoch " << epoch << ": Reconstruction error average: " << (error / batches) << std::endl;

            if(Momentum && epoch == 20){
                momentum = 0.9;
            }

            generate_hidden_images(epoch);
            generate_histograms(epoch);
        }

        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;
    }

    void activate_hidden(vector<double>& h, const vector<double>& v) const {
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

    void activate_visible(const vector<double>& h, vector<double>& v) const {
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
        dbn_assert(end - it == BatchSize, "Invalid size");
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

            for(size_t i = 0; i < num_visible; ++i){
                ga(i) += v1(i) - v2(i);
            }

            for(size_t j = 0; j < num_hidden; ++j){
                gb(j) += h1(j) - h2(j);
            }
        }

        auto n_samples = static_cast<weight>(BatchSize);

        //gw / BatchSize
        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                gw(i, j) /= n_samples;
            }
        }

        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                w_inc(i, j) = w_inc(i, j) * momentum + learning_rate * gw(i,j);
            }
        }

        for(size_t i = 0; i < num_visible; ++i){
            for(size_t j = 0; j < num_hidden; ++j){
                w(i,j) += w_inc(i, j);
            }
        }

        //ga /= BatchSize
        for(size_t i = 0; i < num_visible; ++i){
            ga(i) /= n_samples;
        }

        for(size_t i = 0; i < num_visible; ++i){
            a_inc(i) = a_inc(i) * momentum + learning_rate * ga(i);
        }

        for(size_t i = 0; i < num_visible; ++i){
            a(i) += a_inc(i);
        }

        //gb /= BatchSize
        for(size_t j = 0; j < num_hidden; ++j){
            gb(j) /= n_samples;
        }

        for(size_t j = 0; j < num_hidden; ++j){
            b_inc(j) = b_inc(j) * momentum + learning_rate * gb(j);
        }

        for(size_t j = 0; j < num_hidden; ++j){
            b(j) += b_inc(j);
        }

        //Compute the reconstruction error

        double error = 0.0;
        for(size_t i = 0; i < num_visible; ++i){
            error += ga(i) * ga(i);
        }
        error = sqrt(error / num_visible);

        return error;
    }

    template<typename TrainingItem>
    void reconstruct(const std::vector<TrainingItem>& items){
        dbn_assert(items.size() == num_visible, "The size of the training sample must match visible units");

        //Set the state of the visible units
        for(size_t i = 0; i < num_visible; ++i){
            visibles(i) = items[i];
        }

        activate_hidden(h1, visibles);
        activate_visible(bernoulli(h1, hs), v1);
        bernoulli(v1, visibles);
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

                auto value = w(i,j);
                file << static_cast<size_t>(value > 0 ? static_cast<size_t>(value * 255.0) << 8 : static_cast<size_t>(-value * 255.0) << 16) << " ";
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
        generate_histogram(folder + "/visibles.dat", a.data(), num_visible);
        generate_histogram(folder + "/hiddens.dat", b.data(), num_hidden);

        generate_histogram(folder + "/weights_inc.dat", w_inc.data(), num_visible * num_hidden);
        generate_histogram(folder + "/visibles_inc.dat", a_inc.data(), num_visible);
        generate_histogram(folder + "/hiddens_inc.dat", b_inc.data(), num_hidden);
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