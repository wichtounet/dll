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

    static constexpr const std::size_t NV1 = desc::NV1;
    static constexpr const std::size_t NV2 = desc::NV2;
    static constexpr const std::size_t NH1 = desc::NH1;
    static constexpr const std::size_t NH2 = desc::NH2;
    static constexpr const std::size_t NC = desc::NC;
    static constexpr const std::size_t K = desc::K;

    static constexpr const std::size_t NW1 = NV1 - NH1 + 1; //By definition
    static constexpr const std::size_t NW2 = NV2 - NH2 + 1; //By definition

    etl::fast_matrix<weight, NC, K, NW1, NW2> w;      //shared weights
    etl::fast_vector<weight, K> b;                  //hidden biases bk
    etl::fast_vector<weight, NC> c;                 //visible single bias c

    etl::fast_matrix<weight, NC, NV1, NV2> v1;        //visible units

    etl::fast_matrix<weight, K, NH1, NH2> h1_a;       //Activation probabilities of reconstructed hidden units
    etl::fast_matrix<weight, K, NH1, NH2> h1_s;       //Sampled values of reconstructed hidden units

    etl::fast_matrix<weight, NC, NV1, NV2> v2_a;      //Activation probabilities of reconstructed visible units
    etl::fast_matrix<weight, NC, NV1, NV2> v2_s;      //Sampled values of reconstructed visible units

    etl::fast_matrix<weight, K, NH1, NH2> h2_a;       //Activation probabilities of reconstructed hidden units
    etl::fast_matrix<weight, K, NH1, NH2> h2_s;       //Sampled values of reconstructed hidden units

    //Convolution data

    etl::fast_matrix<weight, 2, K, NH1, NH2> v_cv;    //Temporary convolution
    etl::fast_matrix<weight, 2, NV2, NV2> h_cv;     //Temporary convolution

    conv_rbm() : base_type() {
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        w = 0.01 * etl::normal_generator();
        b = -0.1;
        c = 0.0;
    }

    static constexpr std::size_t input_size() noexcept {
        return NV1 * NV2 * NC;
    }

    static constexpr std::size_t output_size() noexcept {
        return NH1 * NH2 * K;
    }

    static constexpr std::size_t parameters() noexcept {
        return NC * K * NW1 * NW2;
    }

    static std::string to_short_string(){
        char buffer[1024];
        snprintf(buffer, 1024, "CRBM: %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu", NV1, NV2, NC, NW1, NW2, NH1, NH2, K);
        return {buffer};
    }

    void display() const {
        std::cout << to_short_string() << std::endl;
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& v_s){
        activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, v_cv);
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_visible(const H1& h_a, const H2& h_s, V1&& v_a, V2&& v_s){
        activate_visible<P, S>(h_a, h_s, std::forward<V1>(v_a), std::forward<V2>(v_s), h_cv);
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2, typename VCV>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& v_s, VCV&& v_cv){
        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        validate_inputs(v_a, v_s);
        validate_outputs(h_a, h_s);

        using namespace etl;

        v_cv(1) = 0;

        for(std::size_t channel = 0; channel < NC; ++channel){
            for(size_t k = 0; k < K; ++k){
                convolve_2d_valid(v_a(channel), fflip(w(channel)(k)), v_cv(0)(k));
            }

            v_cv(1) += v_cv(0);
        }

        if(P){
            if(hidden_unit == unit_type::BINARY){
                h_a = sigmoid(rep<NH1, NH2>(b) + v_cv(1));
            } else if(hidden_unit == unit_type::RELU){
                h_a = max(rep<NH1, NH2>(b) + v_cv(1), 0.0);
            } else if(hidden_unit == unit_type::RELU6){
                h_a = min(max(rep<NH1, NH2>(b) + v_cv(1), 0.0), 6.0);
            } else if(hidden_unit == unit_type::RELU1){
                h_a = min(max(rep<NH1, NH2>(b) + v_cv(1), 0.0), 1.0);
            }

            nan_check_deep(h_a);
        }

        if(S){
            if(hidden_unit == unit_type::BINARY){
                h_s = bernoulli(h_a);
            } else if(hidden_unit == unit_type::RELU){
                h_s = logistic_noise(h_a);
            } else if(hidden_unit == unit_type::RELU6){
                h_s = ranged_noise(h_a, 6.0);
            } else if(hidden_unit == unit_type::RELU1){
                h_s = ranged_noise(h_a, 1.0);
            }

            nan_check_deep(h_s);
        }
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2, typename HCV>
    void activate_visible(const H1& h_a, const H2& h_s, V1&& v_a, V2&& v_s, HCV&& h_cv){
        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        validate_inputs(v_a, v_s);
        validate_outputs(h_a, h_s);

        using namespace etl;

        for(std::size_t channel = 0; channel < NC; ++channel){
            h_cv(1) = 0.0;

            for(std::size_t k = 0; k < K; ++k){
                convolve_2d_full(h_s(k), w(channel)(k), h_cv(0));
                h_cv(1) += h_cv(0);
            }

            if(P){
                if(visible_unit == unit_type::BINARY){
                    v_a(channel) = sigmoid(c(channel) + h_cv(1));
                } else if(visible_unit == unit_type::GAUSSIAN){
                    v_a(channel) = c(channel) + h_cv(1);
                }
            }

            if(S){
                if(visible_unit == unit_type::BINARY){
                    v_s(channel) = bernoulli(v_a(channel));
                } else if(visible_unit == unit_type::GAUSSIAN){
                    v_s(channel) = normal_noise(v_a(channel));
                }
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

            v_cv(1) = 0;

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(size_t k = 0; k < K; ++k){
                    etl::convolve_2d_valid(v(channel), fflip(w(channel)(k)), v_cv(0)(k));
                }

                v_cv(1) += v_cv(0);
            }

            return - etl::sum(c * etl::sum_r(v)) - etl::sum(b * etl::sum_r(h)) - etl::sum(h * v_cv(1));
        } else if(desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY){
            //Definition according to Honglak Lee / Mixed with Gaussian
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - sum_v ((v - c) ^ 2 / 2)

            v_cv(1) = 0;

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(size_t k = 0; k < K; ++k){
                    etl::convolve_2d_valid(v(channel), fflip(w(channel)(k)), v_cv(0)(k));
                }

                v_cv(1) += v_cv(0);
            }

            return -sum(etl::pow(v - etl::rep<NV1, NV2>(c), 2) / 2.0) - etl::sum(b * etl::sum_r(h)) - etl::sum(h * v_cv(1));
        } else {
            return 0.0;
        }
    }

    template<typename V, typename H, cpp::disable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
    weight energy(const V& v, const H& h){
        static etl::fast_matrix<weight, NC, NV1, NV2> ev;
        static etl::fast_matrix<weight, K, NH1, NH2> eh;

        ev = v;
        eh = h;

        return energy(ev, eh);
    }

    template<typename V>
    weight free_energy_impl(const V& v){
        if(desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY){
            //Definition computed from E(v,h)

            v_cv(1) = 0;

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(size_t k = 0; k < K; ++k){
                    etl::convolve_2d_valid(v(channel), fflip(w(channel)(k)), v_cv(0)(k));
                }

                v_cv(1) += v_cv(0);
            }

            auto x = etl::rep<NH1, NH2>(b) + v_cv(1);

            return - etl::sum(c * etl::sum_r(v)) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else if(desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY){
            //Definition computed from E(v,h)

            v_cv(1) = 0;

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(size_t k = 0; k < K; ++k){
                    etl::convolve_2d_valid(v(channel), fflip(w(channel)(k)), v_cv(0)(k));
                }

                v_cv(1) += v_cv(0);
            }

            auto x = etl::rep<NH1, NH2>(b) + v_cv(1);

            return -sum(etl::pow(v - etl::rep<NV1, NV2>(c), 2) / 2.0) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else {
            return 0.0;
        }
    }

    template<typename V>
    weight free_energy(const V& v){
        static etl::fast_matrix<weight, NC, NV1, NV2> ev;
        ev = v;
        return free_energy_impl(ev);
    }

    weight free_energy() const {
        return free_energy_impl(v1);
    }

    //Utilities for DBNs

    using input_one_t = etl::dyn_matrix<weight, 3>;
    using output_one_t = etl::dyn_matrix<weight, 3>;
    using input_t = std::vector<input_one_t>;
    using output_t = std::vector<output_one_t>;

    template<typename Iterator>
    static auto convert_input(Iterator&& first, Iterator&& last){
        input_t input;
        input.reserve(std::distance(std::forward<Iterator>(first), std::forward<Iterator>(last)));

        std::for_each(std::forward<Iterator>(first), std::forward<Iterator>(last), [&input](auto& sample){
            input.emplace_back(NC, NV1, NV2);
            input.back() = sample;
        });

        return input;
    }

    template<typename Sample>
    static input_one_t convert_sample(const Sample& sample){
        input_one_t result(NC, NV1, NV2);
        result = sample;
        return result;
    }

    output_t prepare_output(std::size_t samples){
        output_t output;
        output.reserve(samples);

        for(std::size_t i = 0; i < samples; ++i){
            output.emplace_back(K, NH1, NH2);
        }

        return output;
    }

    static output_one_t prepare_one_output(){
        return output_one_t(K, NH1, NH2);
    }

    void activate_one(const input_one_t& input, output_one_t& h_a, output_one_t& h_s){
        v1 = input;
        activate_hidden(h_a, h_s, v1, v1);
    }

    void activate_many(const input_t& input, output_t& h_a, output_t& h_s){
        for(std::size_t i = 0; i < input.size(); ++i){
            activate_one(input[i], h_a[i], h_s[i]);
        }
    }

private:
    template<typename V1, typename V2>
    void validate_inputs(const V1& v_a, const V2& v_s){
        static_assert(etl::etl_traits<V1>::dimensions() == 3, "Inputs must be 3D");
        static_assert(etl::etl_traits<V2>::dimensions() == 3, "Inputs must be 3D");

        cpp_assert(etl::dim<0>(v_a) == NC, "Invalid number of input channels");
        cpp_assert(etl::dim<1>(v_a) == NV1, "Invalid input dimensions");
        cpp_assert(etl::dim<2>(v_a) == NV2, "Invalid input dimensions");
        cpp_assert(etl::dim<0>(v_s) == NC, "Invalid number of input channels");
        cpp_assert(etl::dim<1>(v_s) == NV1, "Invalid input dimensions");
        cpp_assert(etl::dim<2>(v_s) == NV2, "Invalid input dimensions");

        cpp_unused(v_a);
        cpp_unused(v_s);
    }

    template<typename H1, typename H2>
    void validate_outputs(const H1& h_a, const H2& h_s){
        static_assert(etl::etl_traits<H1>::dimensions() == 3, "Outputs must be 3D");
        static_assert(etl::etl_traits<H2>::dimensions() == 3, "Outputs must be 3D");

        cpp_assert(etl::dim<0>(h_a) == K, "Invalid number of output channels");
        cpp_assert(etl::dim<1>(h_a) == NH1, "Invalid output dimensions");
        cpp_assert(etl::dim<2>(h_a) == NH2, "Invalid output dimensions");
        cpp_assert(etl::dim<0>(h_s) == K, "Invalid number of output channels");
        cpp_assert(etl::dim<1>(h_s) == NH1, "Invalid output dimensions");
        cpp_assert(etl::dim<2>(h_s) == NH2, "Invalid output dimensions");

        cpp_unused(h_a);
        cpp_unused(h_s);
    }
};

//Allow odr-use of the constexpr static members

template<typename Desc>
const std::size_t conv_rbm<Desc>::NV1;

template<typename Desc>
const std::size_t conv_rbm<Desc>::NV2;

template<typename Desc>
const std::size_t conv_rbm<Desc>::NH1;

template<typename Desc>
const std::size_t conv_rbm<Desc>::NH2;

template<typename Desc>
const std::size_t conv_rbm<Desc>::NC;

template<typename Desc>
const std::size_t conv_rbm<Desc>::NW1;

template<typename Desc>
const std::size_t conv_rbm<Desc>::NW2;

template<typename Desc>
const std::size_t conv_rbm<Desc>::K;

} //end of dll namespace

#endif
