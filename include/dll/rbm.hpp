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
#include <ctime>

//TODO Find a better way to use mkdir
#include <sys/stat.h>

#include "etl/fast_matrix.hpp"
#include "etl/fast_vector.hpp"

#include "assert.hpp"
#include "stop_watch.hpp"
#include "vector.hpp"
#include "batch.hpp"
#include "layer.hpp"
#include "math.hpp"
#include "generic_trainer.hpp"
#include "io.hpp"

namespace dll {

/*!
 * \brief Restricted Boltzmann Machine
 */
template<typename Layer>
class rbm {
public:
    typedef double weight;
    typedef double value_t;

    //TODO Some things, should not be duplicated but used
    //directly from the layer conf

    template<typename RBM>
    using trainer_t = typename Layer::template trainer_t<RBM>;

    static constexpr const std::size_t num_visible = Layer::num_visible;
    static constexpr const std::size_t num_hidden = Layer::num_hidden;

    static constexpr const bool Momentum = Layer::Momentum;
    static constexpr const std::size_t BatchSize = Layer::BatchSize;
    static constexpr const bool Init = Layer::Init;
    static constexpr const bool Debug = Layer::Debug;
    static constexpr const Type VisibleUnit = Layer::VisibleUnit;
    static constexpr const Type HiddenUnit = Layer::HiddenUnit;
    static constexpr const bool DBN = Layer::DBN;
    static constexpr const bool Sparsity = Layer::Sparsity;
    static constexpr const DecayType Decay = Layer::Decay;

    static_assert(BatchSize > 0, "Batch size must be at least 1");

    static_assert(VisibleUnit != Type::SOFTMAX && VisibleUnit != Type::EXP,
        "Exponential and softmax Visible units are not support");
    static_assert(HiddenUnit != Type::GAUSSIAN,
        "Gaussian hidden units are not supported");

    static_assert(!Sparsity || (Sparsity && HiddenUnit == Type::SIGMOID),
        "Sparsity only works with binary hidden units");

    static constexpr const std::size_t num_visible_gra = DBN ? num_visible : 0;
    static constexpr const std::size_t num_hidden_gra = DBN ? num_hidden : 0;

    //Configurable properties
    weight learning_rate =
            VisibleUnit == Type::GAUSSIAN && HiddenUnit == Type::NRLU ? 1e-5
        :   VisibleUnit == Type::GAUSSIAN || HiddenUnit == Type::NRLU ? 1e-4
        :   /* Only NRLU and Gaussian Units needs lower rate */         1e-1;

    weight momentum = 0.5;
    weight weight_cost = 0.0002;

    weight sparsity_target = 0.01;
    weight decay_rate = 0.99;
    weight sparsity_cost = 1.0;

    //Weights and biases
    etl::fast_matrix<weight, num_visible, num_hidden> w;
    etl::fast_vector<weight, num_visible> a;
    etl::fast_vector<weight, num_hidden> b;

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

    std::vector<vector<weight>> gr_probs_a;
    std::vector<vector<weight>> gr_probs_s;

private:
    void init_weights(){
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        static std::default_random_engine rand_engine(std::time(nullptr));
        static std::normal_distribution<weight> distribution(0.0, 1.0);
        static auto generator = std::bind(distribution, rand_engine);

        for(auto& weight : w){
            weight = generator() * 0.1;
        }
    }

public:
    //No copying
    rbm(const rbm& rbm) = delete;
    rbm& operator=(const rbm& rbm) = delete;

    //No moving
    rbm(rbm&& rbm) = delete;
    rbm& operator=(rbm&& rbm) = delete;

    rbm() : a(0.0), b(0.0) {
        init_weights();
    }

    void store(std::ostream& os) const {
        binary_write_all(os, w);
        binary_write_all(os, a);
        binary_write_all(os, b);
    }

    void load(std::istream& is){
        binary_load_all(is, w);
        binary_load_all(is, a);
        binary_load_all(is, b);
    }

    void train(const std::vector<vector<weight>>& training_data, std::size_t max_epochs){
        typedef typename std::remove_reference<decltype(*this)>::type this_type;

        dll::generic_trainer<this_type> trainer;
        trainer.train(*this, training_data, max_epochs);
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
    static void activate_hidden(H& h_a, H& h_s, const V& v_a, const V& v_s, const B& b, const W& w){
        static std::default_random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<weight> normal_distribution(0.0, 1.0);
        static auto normal_generator = std::bind(normal_distribution, rand_engine);

        h_a = 0.0;
        h_s = 0.0;

        if(HiddenUnit == Type::SOFTMAX){
            weight exp_sum = 0.0;

            for(size_t j = 0; j < num_hidden; ++j){
                weight s = 0.0;
                for(size_t i = 0; i < num_visible; ++i){
                    s += w(i, j) * v_a[i];
                }

                auto x = b(j) + s;
                exp_sum += exp(x);
            }

            for(size_t j = 0; j < num_hidden; ++j){
                weight s = 0.0;
                for(size_t i = 0; i < num_visible; ++i){
                    s += w(i, j) * v_a[i];
                }

                auto x = b(j) + s;
                h_a(j) = exp(x) / exp_sum;

                dll_assert(std::isfinite(s), "NaN verify");
                dll_assert(std::isfinite(x), "NaN verify");
                dll_assert(std::isfinite(h_a(j)), "NaN verify");
            }

            std::size_t max_j = 0;
            for(size_t j = 1; j < num_hidden; ++j){
                if(h_a(j) > h_a(max_j)){
                    max_j = j;
                }
            }

            h_s = 0.0;
            h_s(max_j) = 1.0;
        } else {
            for(size_t j = 0; j < num_hidden; ++j){
                weight s = 0.0;
                for(size_t i = 0; i < num_visible; ++i){
                    s += w(i, j) * v_a[i];
                }

                //Total input
                auto x = b(j) + s;

                if(HiddenUnit == Type::SIGMOID){
                    h_a(j) = logistic_sigmoid(x);
                    h_s(j) = h_a(j) > normal_generator() ? 1.0 : 0.0;
                } else if(HiddenUnit == Type::EXP){
                    h_a(j) = exp(x);
                    h_s(j) = h_a(j) > normal_generator() ? 1.0 : 0.0;
                } else if(HiddenUnit == Type::NRLU){
                    std::normal_distribution<weight> noise_distribution(0.0, logistic_sigmoid(x));
                    auto noise = std::bind(noise_distribution, rand_engine);

                    h_a(j) = std::max(0.0, x);
                    h_s(j) = std::max(0.0, x + noise());
                } else {
                    dll_unreachable("Invalid path");
                }

                dll_assert(std::isfinite(s), "NaN verify");
                dll_assert(std::isfinite(x), "NaN verify");
                dll_assert(std::isfinite(h_a(j)), "NaN verify");
                dll_assert(std::isfinite(h_s(j)), "NaN verify");
            }
        }
    }

    template<typename H, typename V>
    void activate_visible(const H& h_a, const H& h_s, V& v_a, V& v_s) const {
        static std::default_random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<weight> normal_distribution(0.0, 1.0);
        static auto normal_generator = std::bind(normal_distribution, rand_engine);

        v_a = 0.0;
        v_s = 0.0;

        for(size_t i = 0; i < num_visible; ++i){
            weight s = 0.0;
            for(size_t j = 0; j < num_hidden; ++j){
                s += w(i, j) * h_s(j);
            }

            //Total input
            auto x = a(i) + s;

            if(VisibleUnit == Type::SIGMOID){
                v_a(i) = logistic_sigmoid(x);
                v_s(i) = v_a(i) > normal_generator() ? 1.0 : 0.0;
            } else if(VisibleUnit == Type::GAUSSIAN){
                std::normal_distribution<weight> noise_distribution(0.0, 1.0);
                auto noise = std::bind(noise_distribution, rand_engine);

                v_a(i) = x;
                v_s(i) = x + noise();
            } else if(VisibleUnit == Type::NRLU){
                std::normal_distribution<weight> noise_distribution(0.0, logistic_sigmoid(x));
                auto noise = std::bind(noise_distribution, rand_engine);

                v_a(i) = std::max(0.0, x);
                v_s(i) = std::max(0.0, x + noise());
            } else {
                dll_unreachable("Invalid path");
            }

            dll_assert(std::isfinite(s), "NaN verify");
            dll_assert(std::isfinite(x), "NaN verify");
            dll_assert(std::isfinite(v_a(i)), "NaN verify");
            dll_assert(std::isfinite(v_s(i)), "NaN verify");
        }
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
        dll_assert(items.size() == num_visible, "The size of the training sample must match visible units");

        stop_watch<> watch;

        //Set the state of the visible units
        v1 = items;

        activate_hidden(h1_a, h1_s, v1, v1);
        activate_visible(h1_a, h1_s, v2_a, v2_s);
        activate_hidden(h2_a, h2_s, v2_a, v2_s);

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