//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_CONV_RBM_INL
#define DLL_CONV_RBM_INL

#include <cstddef>
#include <ctime>
#include <random>

#include "cpp_utils/assert.hpp"             //Assertions
#include "cpp_utils/stop_watch.hpp"         //Performance counter

#include "etl/etl.hpp"
#include "etl/convolution.hpp"

#include "rbm_base.hpp"           //The base class
#include "base_conf.hpp"          //The configuration helpers
#include "math.hpp"               //Logistic sigmoid
#include "io.hpp"                 //Binary load/store functions
#include "tmp.hpp"
#include "rbm_trainer_fwd.hpp"
#include "checks.hpp"

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template<typename Desc>
struct conv_rbm : public rbm_base<Desc> {
    using weight = double;
    using desc = Desc;
    using this_type = conv_rbm<desc>;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static constexpr const std::size_t NV = desc::NV;
    static constexpr const std::size_t NH = desc::NH;
    static constexpr const std::size_t NC = desc::NC;
    static constexpr const std::size_t K = desc::K;

    static constexpr const std::size_t NW = NV - NH + 1; //By definition

    static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN,
        "Only binary and linear visible units are supported");
    static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit),
        "Only binary hidden units are supported");

    etl::fast_matrix<weight, NC, K, NW, NW> w;      //shared weights
    etl::fast_vector<weight, K> b;                  //hidden biases bk
    etl::fast_vector<weight, NC> c;                 //visible single bias c

    etl::fast_matrix<weight, NC, NV, NV> v1;        //visible units

    etl::fast_matrix<weight, K, NH, NH> h1_a;       //Activation probabilities of reconstructed hidden units
    etl::fast_matrix<weight, K, NH, NH> h1_s;       //Sampled values of reconstructed hidden units

    etl::fast_matrix<weight, NC, NV, NV> v2_a;      //Activation probabilities of reconstructed visible units
    etl::fast_matrix<weight, NC, NV, NV> v2_s;      //Sampled values of reconstructed visible units

    etl::fast_matrix<weight, K, NH, NH> h2_a;       //Activation probabilities of reconstructed hidden units
    etl::fast_matrix<weight, K, NH, NH> h2_s;       //Sampled values of reconstructed hidden units

    //Convolution data

    etl::fast_matrix<weight, NC+1, K, NH, NH> v_cv; //Temporary convolution
    etl::fast_matrix<weight, K+1, NV, NV> h_cv;     //Temporary convolution

    //No copying
    conv_rbm(const conv_rbm& rbm) = delete;
    conv_rbm& operator=(const conv_rbm& rbm) = delete;

    //No moving
    conv_rbm(conv_rbm&& rbm) = delete;
    conv_rbm& operator=(conv_rbm&& rbm) = delete;

    conv_rbm(){
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        w = 0.01 * etl::normal_generator();
        b = -0.1;
        c = 0.0;

        //Note: Convolutional RBM needs lower learning rate than standard RBM

        //Better initialization of learning rate
        rbm_base<desc>::learning_rate =
                visible_unit == unit_type::GAUSSIAN  ?             1e-5
            :   is_relu(hidden_unit)                 ?             1e-4
            :   /* Only Gaussian Units needs lower rate */         1e-3;
    }

    static constexpr std::size_t input_size(){
        return NV * NV * NC;
    }

    static constexpr std::size_t output_size(){
        return NH * NH * K;
    }

    void display() const {
        printf("CRBM: %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu\n", NV, NV, NC, NW, NW, NH, NH, K);
    }

    void store(std::ostream& os) const {
        binary_write_all(os, w);
        binary_write_all(os, b);
        binary_write(os, c);
    }

    void load(std::istream& is){
        binary_load_all(is, w);
        binary_load_all(is, b);
        binary_load(is, c);
    }

    template<typename H1, typename H2, typename V1, typename V2>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& v_s){
        activate_hidden(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, v_cv);
    }

    template<typename H1, typename H2, typename V1, typename V2>
    void activate_visible(const H1& h_a, const H2& h_s, V1&& v_a, V2&& v_s){
        activate_visible(h_a, h_s, std::forward<V1>(v_a), std::forward<V2>(v_s), h_cv);
    }

    template<typename H1, typename H2, typename V1, typename V2, typename VCV>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2&, VCV&& v_cv){
        using namespace etl;

        v_cv(NC) = 0;

        for(std::size_t channel = 0; channel < NC; ++channel){
            for(size_t k = 0; k < K; ++k){
                etl::convolve_2d_valid(v_a(channel), fflip(w(channel)(k)), v_cv(channel)(k));
            }

            v_cv(NC) += v_cv(channel);
        }

        if(hidden_unit == unit_type::BINARY){
            h_a = sigmoid(etl::rep<NH, NH>(b) + v_cv(NC));
            h_s = bernoulli(h_a);
        } else if(hidden_unit == unit_type::RELU){
            h_a = max(etl::rep<NH, NH>(b) + v_cv(NC), 0.0);
            h_s = logistic_noise(h_a);
        } else if(hidden_unit == unit_type::RELU6){
            h_a = min(max(etl::rep<NH, NH>(b) + v_cv(NC), 0.0), 6.0);
            h_s = ranged_noise(h_a, 6.0);
        } else if(hidden_unit == unit_type::RELU1){
            h_a = min(max(etl::rep<NH, NH>(b) + v_cv(NC), 0.0), 1.0);
            h_s = ranged_noise(h_a, 1.0);
        } else {
            cpp_unreachable("Invalid path");
        }

        nan_check_deep(h_a);
        nan_check_deep(h_s);
    }

    template<typename H1, typename H2, typename V1, typename V2, typename HCV>
    void activate_visible(const H1&, const H2& h_s, V1&& v_a, V2&& v_s, HCV&& h_cv){
        using namespace etl;

        for(std::size_t channel = 0; channel < NC; ++channel){
            h_cv(K) = 0.0;

            for(std::size_t k = 0; k < K; ++k){
                etl::convolve_2d_full(h_s(k), w(channel)(k), h_cv(k));
                h_cv(K) += h_cv(k);
            }

            if(visible_unit == unit_type::BINARY){
                v_a(channel) = sigmoid(c(channel) + h_cv(K));
                v_s(channel) = bernoulli(v_a(channel));
            } else if(visible_unit == unit_type::GAUSSIAN){
                v_a(channel) = c(channel) + h_cv(K);
                v_s(channel) = normal_noise(v_a(channel));
            } else {
                cpp_unreachable("Invalid path");
            }
        }

        nan_check_deep(v_a);
        nan_check_deep(v_s);
    }

    template<typename Samples, bool EnableWatcher = true, typename RW = void, typename... Args>
    weight train(const Samples& training_data, std::size_t max_epochs, Args... args){
        dll::rbm_trainer<this_type, EnableWatcher, RW> trainer(args...);
        return trainer.train(*this, training_data.begin(), training_data.end(), max_epochs);
    }

    template<typename Iterator, bool EnableWatcher = true, typename RW = void, typename... Args>
    weight train(Iterator&& first, Iterator&& last, std::size_t max_epochs, Args... args){
        dll::rbm_trainer<this_type, EnableWatcher, RW> trainer(args...);
        return trainer.train(*this, std::forward<Iterator>(first), std::forward<Iterator>(last), max_epochs);
    }

    template<typename V>
    weight free_energy(const V&) const {
        weight energy = 0.0;

        //TODO Compute the exact free energy

        return energy;
    }

    weight free_energy() const {
        return free_energy(v1);
    }

    //Utility functions

    template<typename Sample>
    void reconstruct(const Sample& items){
        cpp_assert(items.size() == NV * NV, "The size of the training sample must match visible units");

        cpp::stop_watch<> watch;

        //Set the state of the visible units
        v1 = items;

        activate_hidden(h1_a, h1_s, v1, v1);

        activate_visible(h1_a, h1_s, v2_a, v2_s);
        activate_hidden(h2_a, h2_s, v2_a, v2_s);

        std::cout << "Reconstruction took " << watch.elapsed() << "ms" << std::endl;
    }

    void display_visible_unit_activations() const {
        for(std::size_t channel = 0; channel < NC; ++channel){
            std::cout << "Channel " << channel << std::endl;

            for(size_t i = 0; i < NV; ++i){
                for(size_t j = 0; j < NV; ++j){
                    std::cout << v2_a(channel, i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    void display_visible_unit_samples() const {
        for(std::size_t channel = 0; channel < NC; ++channel){
            std::cout << "Channel " << channel << std::endl;

            for(size_t i = 0; i < NV; ++i){
                for(size_t j = 0; j < NV; ++j){
                    std::cout << v2_s(channel, i, j) << " ";
                }
                std::cout << std::endl;
            }
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