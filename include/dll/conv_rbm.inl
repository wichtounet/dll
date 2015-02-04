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

#include "standard_conv_rbm.hpp"           //The base class
#include "math.hpp"               //Logistic sigmoid
#include "io.hpp"                 //Binary load/store functions
#include "tmp.hpp"
#include "checks.hpp"

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template<typename Desc>
struct conv_rbm final : public standard_conv_rbm<conv_rbm<Desc>, Desc> {
    using desc = Desc;
    using weight = typename desc::weight;
    using this_type = conv_rbm<desc>;
    using base_type = standard_conv_rbm<this_type, desc>;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static constexpr const std::size_t NV = desc::NV;
    static constexpr const std::size_t NH = desc::NH;
    static constexpr const std::size_t NC = desc::NC;
    static constexpr const std::size_t K = desc::K;

    static constexpr const std::size_t NW = NV - NH + 1; //By definition

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

    conv_rbm() : base_type() {
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        w = 0.01 * etl::normal_generator();
        b = -0.1;
        c = 0.0;
    }

    static constexpr std::size_t input_size(){
        return NV * NV * NC;
    }

    static constexpr std::size_t output_size(){
        return NH * NH * K;
    }

    static std::string to_short_string(){
        char buffer[1024];
        snprintf(buffer, 1024, "CRBM: %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu", NV, NV, NC, NW, NW, NH, NH, K);
        return {buffer};
    }

    void display() const {
        std::cout << to_short_string() << std::endl;
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

    template<typename V, typename H, cpp::enable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
    weight energy(const V& v, const H& h){
        if(desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY){
            //Definition according to Honglak Lee
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - c sum_v v

            v_cv(NC) = 0;

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(size_t k = 0; k < K; ++k){
                    etl::convolve_2d_valid(v(channel), fflip(w(channel)(k)), v_cv(channel)(k));
                }

                v_cv(NC) += v_cv(channel);
            }

            return - etl::sum(c * etl::sum_r(v)) - etl::sum(b * etl::sum_r(h)) - etl::sum(h * v_cv(NC));
        } else if(desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY){
            //Definition according to Honglak Lee / Mixed with Gaussian
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - sum_v ((v - c) ^ 2 / 2)

            v_cv(NC) = 0;

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(size_t k = 0; k < K; ++k){
                    etl::convolve_2d_valid(v(channel), fflip(w(channel)(k)), v_cv(channel)(k));
                }

                v_cv(NC) += v_cv(channel);
            }

            return -sum(etl::pow(v - etl::rep<NV, NV>(c), 2) / 2.0) - etl::sum(b * etl::sum_r(h)) - etl::sum(h * v_cv(NC));
        } else {
            return 0.0;
        }
    }

    template<typename V, typename H, cpp::disable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
    weight energy(const V& v, const H& h){
        static etl::fast_matrix<weight, NC, NV, NV> ev;
        static etl::fast_matrix<weight, K, NH, NH> eh;

        ev = v;
        eh = h;

        return energy(ev, eh);
    }

    template<typename V>
    weight free_energy_impl(const V& v){
        if(desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY){
            //Definition computed from E(v,h)

            v_cv(NC) = 0;

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(size_t k = 0; k < K; ++k){
                    etl::convolve_2d_valid(v(channel), fflip(w(channel)(k)), v_cv(channel)(k));
                }

                v_cv(NC) += v_cv(channel);
            }

            auto x = etl::rep<NH, NH>(b) + v_cv(NC);

            return - etl::sum(c * etl::sum_r(v)) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else if(desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY){
            //Definition computed from E(v,h)

            v_cv(NC) = 0;

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(size_t k = 0; k < K; ++k){
                    etl::convolve_2d_valid(v(channel), fflip(w(channel)(k)), v_cv(channel)(k));
                }

                v_cv(NC) += v_cv(channel);
            }

            auto x = etl::rep<NH, NH>(b) + v_cv(NC);

            return -sum(etl::pow(v - etl::rep<NV, NV>(c), 2) / 2.0) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else {
            return 0.0;
        }
    }

    template<typename V>
    weight free_energy(const V& v){
        static etl::fast_matrix<weight, NC, NV, NV> ev;
        ev = v;
        return free_energy_impl(ev);
    }

    weight free_energy() const {
        return free_energy_impl(v1);
    }
};

} //end of dll namespace

#endif
