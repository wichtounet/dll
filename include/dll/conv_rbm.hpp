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

#include "rbm_base.hpp"           //The base class
#include "unit_type.hpp"          //unit_ype enum
#include "assert.hpp"             //Assertions
#include "stop_watch.hpp"         //Performance counter
#include "math.hpp"               //Logistic sigmoid
#include "io.hpp"                 //Binary load/store functions
#include "vector.hpp"             //For samples

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

    template<typename RBM>
    using trainer_t = typename Layer::template trainer_t<RBM>;

    static constexpr const bool Momentum = Layer::Momentum;
    static constexpr const std::size_t BatchSize = Layer::BatchSize;
    static constexpr const Type VisibleUnit = Layer::VisibleUnit;
    static constexpr const Type HiddenUnit = Layer::HiddenUnit;

    static constexpr const std::size_t NV = Layer::NV;
    static constexpr const std::size_t NH = Layer::NH;
    static constexpr const std::size_t K = Layer::K;

    static constexpr const std::size_t NW = NV - NH + 1; //By definition

    static constexpr const std::size_t num_visible = NV * NV;
    static constexpr const std::size_t num_hidden = NH * NH;

    static_assert(VisibleUnit == Type::SIGMOID, "Only binary visible units are supported");
    static_assert(HiddenUnit == Type::SIGMOID, "Only binary hidden units are supported");

    etl::fast_vector<etl::fast_vector<weight, NW * NW>, K> w;     //shared weights
    etl::fast_vector<weight, K> b;                                //hidden biases bk
    weight c;                                                     //visible single bias c

    etl::fast_vector<weight, NV * NV> v1;                         //visible units

    etl::fast_vector<etl::fast_vector<weight, NH * NH>, K> h1_a;  //Activation probabilities of reconstructed hidden units
    etl::fast_vector<etl::fast_vector<weight, NH * NH>, K> h1_s;  //Sampled values of reconstructed hidden units

    etl::fast_vector<weight, NV * NV> v2_a;                       //Activation probabilities of reconstructed visible units
    etl::fast_vector<weight, NV * NV> v2_s;                       //Sampled values of reconstructed visible units

    etl::fast_vector<etl::fast_vector<weight, NH * NH>, K> h2_a;  //Activation probabilities of reconstructed hidden units
    etl::fast_vector<etl::fast_vector<weight, NH * NH>, K> h2_s;  //Sampled values of reconstructed hidden units

    //Convolution data

    etl::fast_vector<etl::fast_vector<weight, NH * NH>, K> v_cv_1;   //Temporary convolution
    etl::fast_vector<etl::fast_vector<weight, NH * NH>, K> v_cv_2;   //Temporary convolution

    etl::fast_vector<etl::fast_vector<weight, NV * NV>, K+1> h_cv_1;   //Temporary convolution
    etl::fast_vector<etl::fast_vector<weight, NV * NV>, K+1> h_cv_2;   //Temporary convolution

    template<typename V, typename K, typename O>
    static void convolve(const V& input, const K& kernel, O& output){
        //TODO Add assertions for the sizes

        //Clear the output
        output = 0.0;

        /*for(std::size_t n = 0 ; n < output.size(); n++){
            for(std::size_t k = 0; k < kernel.size(); ++k){
                output[n] = input[n + k] * kernel[kernel.size() - k - 1];
            }
        }*/

        for(std::size_t i = 0; i < input.size() + kernel.size() + 1; ++i){
            double sum = 0.0;

            auto n_lo = 0 > (i - kernel.size() + 1) ? 0 : i - kernel.size() + 1;
            auto n_hi = input.size() - 1 < i ? input.size() - 1 : i;

            for(std::size_t n = n_lo; n <= n_hi; ++n){
                sum += input[n_lo + n] * kernel[i - n_lo - n];
            }

            output[i] = sum;
        }
    }

public:
    //No copying
    conv_rbm(const conv_rbm& rbm) = delete;
    conv_rbm& operator=(const conv_rbm& rbm) = delete;

    //No moving
    conv_rbm(conv_rbm&& rbm) = delete;
    conv_rbm& operator=(conv_rbm&& rbm) = delete;

    conv_rbm(){}

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
    void activate_hidden(H& h_a, H& h_s, const V& v_a, const V& v_s, CV& v_cv) const {
        static std::default_random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<weight> normal_distribution(0.0, 1.0);
        static auto normal_generator = std::bind(normal_distribution, rand_engine);

        for(size_t k = 0; k < K; ++k){
            h_a(k) = 0.0;
            h_s(k) = 0.0;

            convolve(v_a, w(k), v_cv(k));

            for(size_t j = 0; j < num_hidden; ++j){
                //Total input
                auto x = v_cv(k)(j) + b(k);

                if(HiddenUnit == Type::SIGMOID){
                    h_a(k)(j) = logistic_sigmoid(x);
                    h_s(k)(j) = h_a(k)(j) > normal_generator() ? 1.0 : 0.0;
                } else {
                    dll_unreachable("Invalid path");
                }

                dll_assert(std::isfinite(x), "NaN verify");
                dll_assert(std::isfinite(h_a(k)(j)), "NaN verify");
                dll_assert(std::isfinite(h_s(k)(j)), "NaN verify");
            }
        }
    }

    template<typename H, typename V>
    void activate_visible(const H& h_a, const H& h_s, V& v_a, V& v_s){
        activate_visible(h_a, h_s, v_a, v_s, h_cv_1);
    }

    template<typename H, typename V, typename CV>
    void activate_visible(const H& h_a, const H& h_s, V& v_a, V& v_s, CV& h_cv) const {
        static std::default_random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<weight> normal_distribution(0.0, 1.0);
        static auto normal_generator = std::bind(normal_distribution, rand_engine);

        v_a = 0.0;
        v_s = 0.0;

        for(std::size_t k = 0; k < K; ++k){
            convolve(h_s(k), w(k), h_cv(k));
            h_cv(K) += h_cv(k);
        }

        for(size_t i = 0; i < num_visible; ++i){
            //Total input
            auto x = h_cv(K)(i) + c;

            if(HiddenUnit == Type::SIGMOID){
                v_a(i) = logistic_sigmoid(x);
                v_s(i) = v_a(i) > normal_generator() ? 1.0 : 0.0;
            } else {
                dll_unreachable("Invalid path");
            }

            dll_assert(std::isfinite(x), "NaN verify");
            dll_assert(std::isfinite(v_a(i)), "NaN verify");
            dll_assert(std::isfinite(v_s(i)), "NaN verify");
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

    void display_visible_units(size_t matrix) const {
        for(size_t i = 0; i < matrix; ++i){
            for(size_t j = 0; j < matrix; ++j){
                std::cout << v2_s(i * matrix + j) << " ";
            }
            std::cout << std::endl;
        }
    }
};

} //end of dbn namespace

#endif