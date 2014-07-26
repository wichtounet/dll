//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONV_RBM_INL
#define DBN_CONV_RBM_INL

#include <cstddef>
#include <ctime>
#include <random>

#include "etl/fast_vector.hpp"
#include "etl/dyn_vector.hpp"
#include "etl/fast_matrix.hpp"
#include "etl/convolution.hpp"

#include "rbm_base.hpp"           //The base class
#include "unit_type.hpp"          //unit_ype enum
#include "decay_type.hpp"         //decay_ype enum
#include "assert.hpp"             //Assertions
#include "stop_watch.hpp"         //Performance counter
#include "math.hpp"               //Logistic sigmoid
#include "io.hpp"                 //Binary load/store functions
#include "tmp.hpp"

namespace dll {

template<typename RBM>
struct rbm_trainer;

/*!
 * \brief Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template<typename Desc>
class conv_rbm : public rbm_base<Desc> {
public:
    typedef double weight;
    typedef double value_t;

    using desc = Desc;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static constexpr const std::size_t NV = desc::NV;
    static constexpr const std::size_t NH = desc::NH;
    static constexpr const std::size_t K = desc::K;

    static constexpr const std::size_t NW = NV - NH + 1; //By definition

    static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN,
        "Only binary and linear visible units are supported");
    static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit),
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

    etl::fast_vector<etl::fast_matrix<weight, NH, NH>, K> v_cv;   //Temporary convolution
    etl::fast_vector<etl::fast_matrix<weight, NV, NV>, K+1> h_cv;   //Temporary convolution

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

        for(std::size_t k = 0; k < K; ++k){
            for(auto& weight : w(k)){
                weight = 0.01 * generator();
            }
        }

        b = -0.1;
        c = 0.0;

        //Note: Convolutional RBM needs lower learning rate than standard RBM

        //Better initialization of learning rate
        rbm_base<desc>::learning_rate =
                visible_unit == unit_type::GAUSSIAN  ?             1e-5
            :   is_relu(hidden_unit)                 ?             1e-4
            :   /* Only Gaussian Units needs lower rate */         1e-3;
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
    void activate_hidden(H& h_a, H& h_s, const V& v_a, const V&){
        using namespace etl;

        for(size_t k = 0; k < K; ++k){
            etl::convolve_2d_valid(v_a, fflip(w(k)), v_cv(k));

            if(hidden_unit == unit_type::BINARY){
                h_a(k) = sigmoid(b(k) + v_cv(k));
                h_s(k) = bernoulli(h_a(k));
            } else if(hidden_unit == unit_type::RELU){
                h_a(k) = max(b(k) + v_cv(k), 0.0);
                h_s(k) = logistic_noise(h_a(k));
            } else if(hidden_unit == unit_type::RELU6){
                h_a(k) = min(max(b(k) + v_cv(k), 0.0), 6.0);
                h_s(k) = ranged_noise(h_a(k), 6.0);
            } else if(hidden_unit == unit_type::RELU1){
                h_a(k) = min(max(b(k) + v_cv(k), 0.0), 1.0);
                h_s(k) = ranged_noise(h_a(k), 1.0);
            } else {
                dll_unreachable("Invalid path");
            }

            nan_check_deep(h_a(k));
            nan_check_deep(h_s(k));
        }
    }

    template<typename H, typename V>
    void activate_visible(const H&, const H& h_s, V& v_a, V& v_s){
        using namespace etl;

        h_cv(K) = 0.0;

        for(std::size_t k = 0; k < K; ++k){
            etl::convolve_2d_full(h_s(k), w(k), h_cv(k));
            h_cv(K) += h_cv(k);
        }

        if(visible_unit == unit_type::BINARY){
            v_a = sigmoid(c + h_cv(K));
            v_s = bernoulli(v_a);
        } else if(visible_unit == unit_type::GAUSSIAN){
            v_a = c + h_cv(K);
            v_s = noise(v_a);
        } else {
            dll_unreachable("Invalid path");
        }

        nan_check_deep(v_a);
        nan_check_deep(v_s);
    }

    template<typename Samples>
    weight train(const Samples& training_data, std::size_t max_epochs){
        typedef typename std::remove_reference<decltype(*this)>::type this_type;

        dll::rbm_trainer<this_type> trainer;
        return trainer.train(*this, training_data, max_epochs);
    }

    weight free_energy() const {
        weight energy = 0.0;

        //TODO Compute the exact free energy

        return energy;
    }

    //Utility functions

    template<typename Sample>
    void reconstruct(const Sample& items){
        dll_assert(items.size() == NV * NV, "The size of the training sample must match visible units");

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