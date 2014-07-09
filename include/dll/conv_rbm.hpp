//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONV_RBM_HPP
#define DBN_CONV_RBM_HPP

#include <cstddef>
#include <ctime>
#include <random>

#include "etl/fast_vector.hpp"
#include "etl/fast_matrix.hpp"
#include "etl/convolution.hpp"

#include "rbm_base.hpp"           //The base class
#include "unit_type.hpp"          //unit_ype enum
#include "decay_type.hpp"         //decay_ype enum
#include "assert.hpp"             //Assertions
#include "stop_watch.hpp"         //Performance counter
#include "math.hpp"               //Logistic sigmoid
#include "io.hpp"                 //Binary load/store functions
#include "vector.hpp"             //For samples
#include "tmp.hpp"

namespace dll {

template<typename RBM>
struct generic_trainer;

/*!
 * \brief Convolutional Restricted Boltzmann Machine
 */
template<typename Layer>
class conv_rbm : public rbm_base<Layer> {
public:
    typedef double weight;
    typedef double value_t;

    using layer = Layer;

    static constexpr const unit_type VisibleUnit = Layer::VisibleUnit;
    static constexpr const unit_type HiddenUnit = Layer::HiddenUnit;

    static constexpr const std::size_t NV = Layer::NV;
    static constexpr const std::size_t NH = Layer::NH;
    static constexpr const std::size_t K = Layer::K;

    static constexpr const std::size_t NW = NV - NH + 1; //By definition

    static constexpr const std::size_t num_visible = NV * NV;
    static constexpr const std::size_t num_hidden = NH * NH;

    static_assert(VisibleUnit == unit_type::BINARY || VisibleUnit == unit_type::GAUSSIAN,
        "Only binary and linear visible units are supported");
    static_assert(HiddenUnit == unit_type::BINARY,
        "Only binary hidden units are supported");

    etl::fast_vector<etl::fast_matrix<weight, NW, NW>, K> w;      //shared weights
    etl::fast_vector<weight, K> b;                                //hidden biases bk
    weight c;                                                     //visible single bias c

    etl::fast_matrix<weight, NV, NV> v1;                         //visible units

    etl::fast_vector<etl::fast_matrix<weight, NH, NH>, K> h1_a;  //Activation probabilities of reconstructed hidden units
    etl::fast_vector<etl::fast_matrix<weight, NH, NH>, K> h1_s;  //Sampled values of reconstructed hidden units

    etl::fast_matrix<weight, NV, NV> v2_a;                       //Activation probabilities of reconstructed visible units
    etl::fast_matrix<weight, NV, NV> v2_s;                       //Sampled values of reconstructed visible units

    etl::fast_vector<etl::fast_matrix<weight, NH, NH>, K> h2_a;  //Activation probabilities of reconstructed hidden units
    etl::fast_vector<etl::fast_matrix<weight, NH, NH>, K> h2_s;  //Sampled values of reconstructed hidden units

    //Convolution data

    etl::fast_vector<etl::fast_matrix<weight, NH, NH>, K> v_cv_1;   //Temporary convolution
    etl::fast_vector<etl::fast_matrix<weight, NH, NH>, K> v_cv_2;   //Temporary convolution

    etl::fast_vector<etl::fast_matrix<weight, NV, NV>, K+1> h_cv_1;   //Temporary convolution
    etl::fast_vector<etl::fast_matrix<weight, NV, NV>, K+1> h_cv_2;   //Temporary convolution

public:
    //No copying
    conv_rbm(const conv_rbm& rbm) = delete;
    conv_rbm& operator=(const conv_rbm& rbm) = delete;

    //No moving
    conv_rbm(conv_rbm&& rbm) = delete;
    conv_rbm& operator=(conv_rbm&& rbm) = delete;

    conv_rbm(){
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        static std::default_random_engine rand_engine(std::time(nullptr));
        static std::normal_distribution<weight> distribution(0.0, 1.0);
        static auto generator = std::bind(distribution, rand_engine);

        double scale = 0.001;

        for(std::size_t k = 0; k < K; ++k){
            for(auto& weight : w(k)){
                weight = scale * generator();
            }
        }

        for(std::size_t k = 0; k < K; ++k){
            b(k) = 2 * scale * generator();
        }

        c = scale * generator();
    }

    void store(std::ostream& os) const {
        for(std::size_t k = 0; k < K; ++k){
            binary_write_all(os, w(k));
        }
        binary_write_all(os, b);
        binary_write(os, c);
    }

    void load(std::istream& is){
        for(std::size_t k = 0; k < K; ++k){
            binary_load_all(is, w(k));
        }
        binary_load_all(is, b);
        binary_load(is, c);
    }

    template<typename H, typename V>
    void activate_hidden(H& h_a, H& h_s, const V& v_a, const V& v_s){
        activate_hidden(h_a, h_s, v_a, v_s, v_cv_1);
    }

    template<typename H, typename V, typename CV>
    void activate_hidden(H& h_a, H& h_s, const V& v_a, const V&, CV& v_cv){
        static std::default_random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<weight> normal_distribution(0.0, 1.0);
        static auto normal_generator = std::bind(normal_distribution, rand_engine);

        for(size_t k = 0; k < K; ++k){
            h_a(k) = 0.0;
            h_s(k) = 0.0;

            std::reverse(w(k).begin(), w(k).end());
            etl::convolve_2d_valid(v_a, w(k), v_cv(k));
            std::reverse(w(k).begin(), w(k).end());

            for(size_t i = 0; i < NH; ++i){
                for(size_t j = 0; j < NH; ++j){
                    //Total input
                    auto x = v_cv(k)(i,j) + b(k);

                    if(HiddenUnit == unit_type::BINARY){
                        h_a(k)(i,j) = logistic_sigmoid(x);
                        h_s(k)(i,j) = h_a(k)(i,j) > normal_generator() ? 1.0 : 0.0;
                    } else {
                        dll_unreachable("Invalid path");
                    }

                    dll_assert(std::isfinite(x), "NaN verify");
                    dll_assert(std::isfinite(h_a(k)(i,j)), "NaN verify");
                    dll_assert(std::isfinite(h_s(k)(i,j)), "NaN verify");
                }
            }
        }
    }

    template<typename H, typename V>
    void activate_visible(const H& h_a, const H& h_s, V& v_a, V& v_s){
        activate_visible(h_a, h_s, v_a, v_s, h_cv_1);
    }

    template<typename H, typename V, typename CV>
    void activate_visible(const H&, const H& h_s, V& v_a, V& v_s, CV& h_cv) const {
        static std::default_random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<weight> normal_distribution(0.0, 1.0);
        static auto normal_generator = std::bind(normal_distribution, rand_engine);

        v_a = 0.0;
        v_s = 0.0;

        for(std::size_t k = 0; k < K; ++k){
            etl::convolve_2d_full(h_s(k), w(k), h_cv(k));
            h_cv(K) += h_cv(k);
        }

        for(size_t i = 0; i < NV; ++i){
            for(size_t j = 0; j < NV; ++j){
                //Total input
                auto x = h_cv(K)(i,j) + c;

                if(VisibleUnit == unit_type::BINARY){
                    v_a(i,j) = logistic_sigmoid(x);
                    v_s(i,j) = v_a(i,j) > normal_generator() ? 1.0 : 0.0;
                } else if(VisibleUnit == unit_type::GAUSSIAN){
                    std::normal_distribution<weight> noise_distribution(0.0, 1.0);
                    auto noise = std::bind(noise_distribution, rand_engine);

                    v_a(i,j) = x;
                    v_s(i,j) = x + noise();
                } else {
                    dll_unreachable("Invalid path");
                }

                dll_assert(std::isfinite(x), "NaN verify");
                dll_assert(std::isfinite(v_a(i,j)), "NaN verify");
                dll_assert(std::isfinite(v_s(i,j)), "NaN verify");
            }
        }
    }

    void train(const std::vector<vector<weight>>& training_data, std::size_t max_epochs){
        typedef typename std::remove_reference<decltype(*this)>::type this_type;

        dll::generic_trainer<this_type> trainer;
        trainer.train(*this, training_data, max_epochs);
    }

    weight free_energy() const {
        weight energy = 0.0;

        //TODO Compute the exact free energy

        return energy;
    }

    //Utility functions

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

    void display_visible_unit_activations() const {
        for(size_t i = 0; i < NV; ++i){
            for(size_t j = 0; j < NV; ++j){
                std::cout << v2_a(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    void display_visible_unit_samples() const {
        for(size_t i = 0; i < NV; ++i){
            for(size_t j = 0; j < NV; ++j){
                std::cout << v2_s(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    void display_hidden_unit_activations() const {
        for(size_t k = 0; k < K; ++k){
            for(size_t i = 0; i < NV; ++i){
                for(size_t j = 0; j < NV; ++j){
                    std::cout << h2_a(k)(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl << std::endl;
        }
    }

    void display_hidden_unit_samples() const {
        for(size_t k = 0; k < K; ++k){
            for(size_t i = 0; i < NV; ++i){
                for(size_t j = 0; j < NV; ++j){
                    std::cout << h2_s(k)(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl << std::endl;
        }
    }
};

} //end of dbn namespace

#endif