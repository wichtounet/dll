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
#include <cmath>

//TODO Find a better way to use mkdir
#include <sys/stat.h>

#include "assert.hpp"
#include "stop_watch.hpp"
#include "fast_matrix.hpp"
#include "fast_vector.hpp"
#include "vector.hpp"
#include "conf.hpp"
#include "batch.hpp"

#ifdef NDEBUG
#define nan_check(list)
#else
#define nan_check(list) for(auto& nantest : ((list))){dbn_assert(std::isfinite(nantest), "NaN Verify");}
#endif

namespace dbn {

/*!
 * \brief Restricted Boltzmann Machine
 */
template<typename Layer>
class rbm {
public:
    typedef double weight;
    typedef double value_t;

    static constexpr const std::size_t num_visible = Layer::num_visible;
    static constexpr const std::size_t num_hidden = Layer::num_hidden;

    static constexpr const bool Momentum = Layer::Conf::Momentum;
    static constexpr const std::size_t BatchSize = Layer::Conf::BatchSize;
    static constexpr const bool Init = Layer::Conf::Debug;
    static constexpr const bool Debug = Layer::Conf::Debug;
    static constexpr const Type VisibleUnit = Layer::Conf::VisibleUnit;
    static constexpr const Type HiddenUnit = Layer::Conf::HiddenUnit;
    static constexpr const bool DBN = Layer::Conf::DBN;
    static constexpr const bool Decay = Layer::Conf::Decay;

    static_assert(BatchSize > 0, "Batch size must be at least 1");

    static_assert(VisibleUnit == Type::SIGMOID || VisibleUnit == Type::GAUSSIAN,
        "Only logistic and gaussian visible units are supported");
    static_assert(HiddenUnit != Type::GAUSSIAN,
        "Gaussian hidden units are not supported");

    static constexpr const std::size_t num_visible_mom = Momentum ? num_visible : 0;
    static constexpr const std::size_t num_hidden_mom = Momentum ? num_hidden : 0;

    static constexpr const std::size_t num_visible_gra = DBN ? num_visible : 0;
    static constexpr const std::size_t num_hidden_gra = DBN ? num_hidden : 0;

private:
    fast_vector<value_t, num_visible> visibles;
    fast_vector<value_t, num_hidden> hiddens;

    fast_matrix<weight, num_visible, num_hidden> w;
    fast_vector<weight, num_visible> a;
    fast_vector<weight, num_hidden> b;

    //Weights for momentum
    fast_matrix<weight, num_visible_mom, num_hidden_mom> w_inc;
    fast_vector<weight, num_visible_mom> a_inc;
    fast_vector<weight, num_hidden_mom> b_inc;

    //Temporary data
    //TODO Perhaps it is better as static data in the functions
    fast_vector<weight, num_visible> v1;
    fast_vector<weight, num_hidden> h1;
    fast_vector<weight, num_visible> v2;
    fast_vector<weight, num_hidden> h2;

    //Deltas
    fast_matrix<weight, num_visible, num_hidden> gw;
    fast_vector<weight, num_visible> ga;
    fast_vector<weight, num_hidden> gb;

public:
    //Gradients computations for DBN
    fast_matrix<weight, num_visible, num_hidden>& gr_w = w;
    fast_vector<weight, num_hidden>& gr_b = b;

    fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_incs;
    fast_vector<weight, num_hidden_gra> gr_b_incs;

    fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_best;
    fast_vector<weight, num_hidden_gra> gr_b_best;

    fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_best_incs;
    fast_vector<weight, num_hidden_gra> gr_b_best_incs;

    fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_df0;
    fast_vector<weight, num_hidden_gra> gr_b_df0;

    fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_df3;
    fast_vector<weight, num_hidden_gra> gr_b_df3;

    fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_s;
    fast_vector<weight, num_hidden_gra> gr_b_s;

    fast_matrix<weight, num_visible_gra, num_hidden_gra> gr_w_tmp;
    fast_vector<weight, num_hidden_gra> gr_b_tmp;

    std::vector<vector<weight>> gr_probs;

private:
    //TODO Add a way to configure that
    weight learning_rate = VisibleUnit == Type::SIGMOID ? 0.1 : 0.0001;
    weight momentum = 0.5;
    weight weight_cost = 0.0002;

    void init_weights(){
        //Initialize the weights using a Gaussian distribution of mean 0 and
        //variance 0.0.1
        static std::default_random_engine rand_engine(::time(nullptr));
        static std::normal_distribution<weight> distribution(0.0, 1.0);
        static auto generator = std::bind(distribution, rand_engine);

        for(auto& weight : w){
            weight = generator() * 0.1;
        }
    }

    static constexpr weight logistic_sigmoid(weight x){
        return 1.0 / (1.0 + exp(-x));
    }

public:
    //No copying
    rbm(const rbm& rbm) = delete;
    rbm& operator=(const rbm& rbm) = delete;

    //No moving
    rbm(rbm&& rbm) = delete;
    rbm& operator=(rbm&& rbm) = delete;

    template<bool M = Momentum, typename std::enable_if<(!M), bool>::type = false>
    rbm() : a(0.0), b(0.0){
        static_assert(!Momentum, "This constructor should only be used without momentum support");

        init_weights();
    }

    template<bool M = Momentum, typename std::enable_if<(M), bool>::type = false>
    rbm() : a(0.0), b(0.0), w_inc(0.0), a_inc(0.0), b_inc(0.0) {
        static_assert(Momentum, "This constructor should only be used with momentum support");

        init_weights();
    }

    template<typename T>
    static void binary_write(std::ostream& os, const T& v){
        os.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    template<typename Container>
    static void binary_write_all(std::ostream& os, const Container& c){
        for(auto& v : c){
            binary_write(os, v);
        }
    }

    void store(std::ostream& os) const {
        binary_write_all(os, w);
        binary_write_all(os, a);
        binary_write_all(os, b);
    }

    template<typename T>
    static void binary_load(std::istream& is, T& v){
        is.read(reinterpret_cast<char*>(&v), sizeof(v));
    }

    template<typename Container>
    static void binary_load_all(std::istream& is, Container& c){
        for(auto& v : c){
            binary_load(is, v);
        }
    }

    void load(std::istream& is){
        binary_load_all(is, w);
        binary_load_all(is, a);
        binary_load_all(is, b);
    }

    void train(const std::vector<vector<weight>>& training_data, std::size_t max_epochs){
        stop_watch<std::chrono::seconds> watch;

        if(Init){
            //Initialize the visible biases to log(pi/(1-pi))
            for(size_t i = 0; i < num_visible; ++i){
                size_t c = 0;
                for(auto& items : training_data){
                    if(items[i] == 1){
                        ++c;
                    }
                }

                auto pi = static_cast<weight>(c) / training_data.size();
                pi += 0.0001;
                a(i) = log(pi / (1.0 - pi));

                dbn_assert(std::isfinite(a(i)), "NaN verify");
            }
        }

        auto batches = training_data.size() / BatchSize + (training_data.size() % BatchSize == 0 ? 0 : 1);

        for(size_t epoch= 0; epoch < max_epochs; ++epoch){
            weight error = 0.0;
            for(size_t i = 0; i < batches; ++i){
                auto start = i * BatchSize;
                auto end = std::min(start + BatchSize, training_data.size());

                error += cd_step(dbn::batch<vector<weight>>(training_data.begin() + start, training_data.begin() + end));
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
        return activate_hidden(h, v, b, w);
    }

    template<bool Temp, typename V1, typename V2>
    void gr_activate_hidden(V1& h, const V2& v) const {
        return activate_hidden(h, v, Temp ? gr_b_tmp : gr_b, Temp ? gr_w_tmp : gr_w);
    }

    template<typename V1, typename V2, typename V3, typename V4>
    static void activate_hidden(V1& h, const V2& v, const V3& b, const V4& w){
        h = 0.0;

        if(HiddenUnit == Type::SOFTMAX){
            weight exp_sum = 0.0;

            for(size_t j = 0; j < num_hidden; ++j){
                weight s = 0.0;
                for(size_t i = 0; i < num_visible; ++i){
                    s += w(i, j) * v[i];
                }

                auto x = b(j) + s;
                exp_sum += exp(x);
            }

            for(size_t j = 0; j < num_hidden; ++j){
                weight s = 0.0;
                for(size_t i = 0; i < num_visible; ++i){
                    s += w(i, j) * v[i];
                }

                auto x = b(j) + s;
                h(j) = exp(x) / exp_sum;

                dbn_assert(std::isfinite(s), "NaN verify");
                dbn_assert(std::isfinite(x), "NaN verify");
                dbn_assert(std::isfinite(h(j)), "NaN verify");
            }
        } else {
            for(size_t j = 0; j < num_hidden; ++j){
                weight s = 0.0;
                for(size_t i = 0; i < num_visible; ++i){
                    s += w(i, j) * v[i];
                }

                auto x = b(j) + s;
                if(HiddenUnit == Type::SIGMOID){
                    h(j) = logistic_sigmoid(x);
                } else if(HiddenUnit == Type::EXP){
                    h(j) = exp(x);
                }

                dbn_assert(std::isfinite(s), "NaN verify");
                dbn_assert(std::isfinite(x), "NaN verify");
                dbn_assert(std::isfinite(h(j)), "NaN verify");
            }
        }
    }

    template<typename V1, typename V2>
    void activate_visible(const V1& h, V2& v) const {
        v = 0.0;

        static std::default_random_engine rand_engine(::time(nullptr));
        static std::uniform_real_distribution<weight> distribution(0.0, 1.0);
        static auto generator = bind(distribution, rand_engine);

        auto bernoulli = [](weight v){ return generator() < v ? 1.0 : 0.0; };
        auto identity = [](weight v){ return v; };

        auto ht = VisibleUnit == Type::SIGMOID ? bernoulli : identity;

        for(size_t i = 0; i < num_visible; ++i){
            weight s = 0.0;
            for(size_t j = 0; j < num_hidden; ++j){
                s += w(i, j) * ht(h(j));
            }

            auto activation = a(i) + s;

            if(VisibleUnit == Type::SIGMOID){
                v(i) = logistic_sigmoid(activation);
            } else if(VisibleUnit == Type::GAUSSIAN){
                v(i) = activation;
            }

            dbn_assert(std::isfinite(s), "NaN verify");
            dbn_assert(std::isfinite(activation), "NaN verify");
            dbn_assert(std::isfinite(v(i)), "NaN verify");
        }
    }

    template<typename T>
    weight cd_step(const dbn::batch<T> batch){
        dbn_assert(batch.size() <= static_cast<typename dbn::batch<T>::size_type>(BatchSize), "Invalid size");
        dbn_assert(batch[0].size() == num_visible, "The size of the training sample must match visible units");

        v1 = 0.0;
        h1 = 0.0;
        v2 = 0.0;
        h2 = 0.0;

        ga = 0.0;
        gb = 0.0;
        gw = 0.0;

        for(auto& items : batch){
            for(size_t i = 0; i < num_visible; ++i){
                v1(i) = items[i];
            }

            activate_hidden(h1, v1);
            activate_visible(h1, v2);
            activate_hidden(h2, v2);

            for(size_t i = 0; i < num_visible; ++i){
                for(size_t j = 0; j < num_hidden; ++j){
                    gw(i, j) += h1(j) * v1(i) - h2(j) * v2(i);
                }
            }

            ga += v1 - v2;
            gb += h1 - h2;
        }

        nan_check(gw);

        auto n_samples = static_cast<weight>(batch.size());

        if(Momentum){
            if(Decay){
                w_inc = w_inc * momentum + ((gw / n_samples) - (w * weight_cost)) * learning_rate;
            } else {
                w_inc = w_inc * momentum + gw * (learning_rate / n_samples);
            }

            w += w_inc;
        } else {
            if(Decay){
                w += ((gw / n_samples) - (w * weight_cost)) * learning_rate;
            } else {
                w += (gw / n_samples) * learning_rate;
            }
        }

        nan_check(w);

        if(Momentum){
            a_inc = a_inc * momentum + (ga  / n_samples) * learning_rate;
            a += a_inc;
        } else {
            a += (ga / n_samples) * learning_rate;
        }

        nan_check(a);

        if(Momentum){
            b_inc = b_inc * momentum + (gb / n_samples) * learning_rate;
            b += b_inc;
        } else {
            b += (gb / n_samples) * learning_rate;
        }

        nan_check(b);

        //Compute the reconstruction error

        weight error = 0.0;
        for(size_t i = 0; i < num_visible; ++i){
            error += ga(i) * ga(i);
        }
        error = sqrt((error / (n_samples * n_samples)) / num_visible);

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

    void reconstruct(const vector<weight>& items){
        dbn_assert(items.size() == num_visible, "The size of the training sample must match visible units");

        stop_watch<> watch;

        //Set the state of the visible units
        for(size_t i = 0; i < num_visible; ++i){
            visibles(i) = items[i];
        }

        activate_hidden(h1, visibles);
        activate_visible(h1, v1);
        activate_hidden(v1, visibles);

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

        generate_histogram(folder + "/weights.dat", w);
        generate_histogram(folder + "/visibles.dat", a);
        generate_histogram(folder + "/hiddens.dat", b);

        if(Momentum){
            generate_histogram(folder + "/weights_inc.dat", w_inc);
            generate_histogram(folder + "/visibles_inc.dat", a_inc);
            generate_histogram(folder + "/hiddens_inc.dat", b_inc);
        }
    }

    template<typename Container>
    void generate_histogram(const std::string& path, const Container& weights){
        std::ofstream file(path, std::ios::out);

        if(!file){
            std::cout << "Could not open file " << path << std::endl;
        } else {
            for(auto& weight : weights){
                file << weight << std::endl;
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