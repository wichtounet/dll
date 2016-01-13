//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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
#include "cpp_utils/maybe_parallel.hpp"

#include "etl/etl.hpp"

#include "standard_conv_rbm.hpp"  //The base class
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

    static constexpr const bool dbn_only = layer_traits<this_type>::is_dbn_only();
    static constexpr const bool memory = layer_traits<this_type>::is_memory();

    template<std::size_t B>
    using input_batch_t = etl::fast_dyn_matrix<weight, B, NC, NV1, NV2>;

    template<std::size_t B>
    using output_batch_t = etl::fast_dyn_matrix<weight, B, K, NH1, NH2>;

    using w_type = etl::fast_matrix<weight, NC, K, NW1, NW2>;
    using b_type = etl::fast_vector<weight, K>;
    using c_type = etl::fast_vector<weight, NC>;

    w_type w;      //!< shared weights
    b_type b;      //!< hidden biases bk
    c_type c;      //!< visible single bias c

    std::unique_ptr<w_type> bak_w;      //!< backup shared weights
    std::unique_ptr<b_type> bak_b;      //!< backup hidden biases bk
    std::unique_ptr<c_type> bak_c;      //!< backup visible single bias c

    etl::fast_matrix<weight, NC, NV1, NV2> v1;        //visible units

    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h1_a;       //Activation probabilities of reconstructed hidden units
    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h1_s;       //Sampled values of reconstructed hidden units

    conditional_fast_matrix_t<!dbn_only, weight, NC, NV1, NV2> v2_a;      //Activation probabilities of reconstructed visible units
    conditional_fast_matrix_t<!dbn_only, weight, NC, NV1, NV2> v2_s;      //Sampled values of reconstructed visible units

    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h2_a;       //Activation probabilities of reconstructed hidden units
    conditional_fast_matrix_t<!dbn_only, weight, K, NH1, NH2> h2_s;       //Sampled values of reconstructed hidden units

    //Convolution data

    static constexpr const std::size_t V_CV_CHANNELS = 2;
    static constexpr const std::size_t H_CV_CHANNELS = 2;

    //Note: These are used by activation functions and therefore are
    //needed in dbn_only mode as well
    etl::fast_matrix<weight, V_CV_CHANNELS, K, NH1, NH2> v_cv;      //Temporary convolution
    etl::fast_matrix<weight, H_CV_CHANNELS, NV2, NV2> h_cv;         //Temporary convolution

    mutable cpp::thread_pool<!layer_traits<this_type>::is_serial()> pool;

    conv_rbm() : base_type() {
        if(is_relu(hidden_unit)){
            w = etl::normal_generator(0.0, 0.01);
            b = 0.0;
            c = 0.0;
        } else {
            w = 0.01 * etl::normal_generator();
            b = -0.1;
            c = 0.0;
        }
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

    void backup_weights(){
        unique_safe_get(bak_w) = w;
        unique_safe_get(bak_b) = b;
        unique_safe_get(bak_c) = c;
    }

    void restore_weights(){
        w = *bak_w;
        b = *bak_b;
        c = *bak_c;
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& v_s) const {
        etl::fast_dyn_matrix<weight, V_CV_CHANNELS, K, NH1, NH2> v_cv;      //Temporary convolution
        activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, v_cv);
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_visible(const H1& h_a, const H2& h_s, V1&& v_a, V2&& v_s) const {
        etl::fast_dyn_matrix<weight, H_CV_CHANNELS, NV2, NV2> h_cv;         //Temporary convolution
        activate_visible<P, S>(h_a, h_s, std::forward<V1>(v_a), std::forward<V2>(v_s), h_cv);
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2, typename VCV>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& /*v_s*/, VCV&& v_cv) const {
        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        validate_inputs<V1,V2>();
        validate_outputs<H1,H2>();

        using namespace etl;

        base_type::template compute_vcv<this_type>(v_a, v_cv, w);

        if(hidden_unit == unit_type::BINARY){
            if(visible_unit == unit_type::BINARY){
                h_a = sigmoid(rep<NH1, NH2>(b) + v_cv(1));
            } else if(visible_unit == unit_type::GAUSSIAN){
                h_a = sigmoid((1.0 / (0.1 * 0.1)) >> (rep<NH1, NH2>(b) + v_cv(1)));
            }
        } else if(hidden_unit == unit_type::RELU){
            h_a = max(rep<NH1, NH2>(b) + v_cv(1), 0.0);
        } else if(hidden_unit == unit_type::RELU6){
            h_a = min(max(rep<NH1, NH2>(b) + v_cv(1), 0.0), 6.0);
        } else if(hidden_unit == unit_type::RELU1){
            h_a = min(max(rep<NH1, NH2>(b) + v_cv(1), 0.0), 1.0);
        }

        nan_check_deep(h_a);

        if(S){
            if(hidden_unit == unit_type::BINARY){
                h_s = bernoulli(h_a);
            } else if(hidden_unit == unit_type::RELU){
                h_s = max(logistic_noise(rep<NH1, NH2>(b) + v_cv(1)), 0.0);
            } else if(hidden_unit == unit_type::RELU6){
                h_s = ranged_noise(h_a, 6.0);
            } else if(hidden_unit == unit_type::RELU1){
                h_s = ranged_noise(h_a, 1.0);
            }

            nan_check_deep(h_s);
        }
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2, typename HCV>
    void activate_visible(const H1& /*h_a*/, const H2& h_s, V1&& v_a, V2&& v_s, HCV&& h_cv) const {
        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        validate_inputs<V1,V2>();
        validate_outputs<H1,H2>();

        using namespace etl;

        base_type::template compute_hcv<this_type>(h_s, h_cv, w, [&](std::size_t channel){
            if(visible_unit == unit_type::BINARY){
                v_a(channel) = sigmoid(c(channel) + h_cv(1));
            } else if(visible_unit == unit_type::GAUSSIAN){
                v_a(channel) = c(channel) + h_cv(1);
            }
        });

        nan_check_deep(v_a);

        if(S){
            if(visible_unit == unit_type::BINARY){
                v_s = bernoulli(v_a);
            } else if(visible_unit == unit_type::GAUSSIAN){
                v_s = normal_noise(v_a);
            }

            nan_check_deep(v_s);
        }
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2, typename VCV>
    void batch_activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& /*v_s*/, VCV&& v_cv) const {
        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        static_assert(etl::decay_traits<H1>::template dim<0>() == etl::decay_traits<H2>::template dim<0>(), "Inconsistent number of batches");
        static_assert(etl::decay_traits<H1>::template dim<0>() == etl::decay_traits<V1>::template dim<0>(), "Inconsistent number of batches");
        static_assert(etl::decay_traits<H1>::template dim<0>() == etl::decay_traits<V2>::template dim<0>(), "Inconsistent number of batches");
        static_assert(etl::decay_traits<H1>::template dim<0>() == etl::decay_traits<VCV>::template dim<0>(), "Inconsistent number of batches");

        validate_inputs<V1,V2,1>();
        validate_outputs<H1,H2,1>();

        using namespace etl;

        base_type::template batch_compute_vcv<this_type>(pool, v_a, v_cv, w, [&](std::size_t batch){
            if(hidden_unit == unit_type::BINARY){
                if(visible_unit == unit_type::BINARY){
                    h_a(batch) = sigmoid(rep<NH1, NH2>(b) + v_cv(batch)(1));
                } else if(visible_unit == unit_type::GAUSSIAN){
                    h_a(batch) = sigmoid((1.0 / (0.1 * 0.1)) >> (rep<NH1, NH2>(b) + v_cv(batch)(1)));
                }
            } else if(hidden_unit == unit_type::RELU){
                h_a(batch) = max(rep<NH1, NH2>(b) + v_cv(batch)(1), 0.0);

                if(S){
                    h_s(batch) = max(logistic_noise(rep<NH1, NH2>(b) + v_cv(batch)(1)), 0.0);
                }
            } else if(hidden_unit == unit_type::RELU6){
                h_a(batch) = min(max(rep<NH1, NH2>(b) + v_cv(batch)(1), 0.0), 6.0);
            } else if(hidden_unit == unit_type::RELU1){
                h_a(batch) = min(max(rep<NH1, NH2>(b) + v_cv(batch)(1), 0.0), 1.0);
            }
        });

        nan_check_deep(h_a);

        if(S){
            if(hidden_unit == unit_type::BINARY){
                h_s = bernoulli(h_a);
            } else if(hidden_unit == unit_type::RELU6){
                h_s = ranged_noise(h_a, 6.0);
            } else if(hidden_unit == unit_type::RELU1){
                h_s = ranged_noise(h_a, 1.0);
            }

            nan_check_deep(h_s);
        }
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2, typename HCV>
    void batch_activate_visible(const H1& /*h_a*/, const H2& h_s, V1&& v_a, V2&& v_s, HCV&& h_cv) const {
        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        static_assert(etl::decay_traits<H1>::template dim<0>() == etl::decay_traits<H2>::template dim<0>(), "Inconsistent number of batches");
        static_assert(etl::decay_traits<H1>::template dim<0>() == etl::decay_traits<V1>::template dim<0>(), "Inconsistent number of batches");
        static_assert(etl::decay_traits<H1>::template dim<0>() == etl::decay_traits<V2>::template dim<0>(), "Inconsistent number of batches");
        static_assert(etl::decay_traits<H1>::template dim<0>() == etl::decay_traits<HCV>::template dim<0>(), "Inconsistent number of batches");

        validate_inputs<V1,V2,1>();
        validate_outputs<H1,H2,1>();

        base_type::template batch_compute_hcv<this_type>(pool, h_s, h_cv, w, [&](std::size_t batch, std::size_t channel){
            if(visible_unit == unit_type::BINARY){
                v_a(batch)(channel) = etl::sigmoid(c(channel) + h_cv(batch)(1));
            } else if(visible_unit == unit_type::GAUSSIAN){
                v_a(batch)(channel) = c(channel) + h_cv(batch)(1);
            }
        });

        nan_check_deep(v_a);

        if(S){
            if(visible_unit == unit_type::BINARY){
                v_s = bernoulli(v_a);
            } else if(visible_unit == unit_type::GAUSSIAN){
                v_s = normal_noise(v_a);
            }

            nan_check_deep(v_s);
        }
    }

    template<typename V, typename H, cpp_enable_if(etl::is_etl_expr<V>::value)>
    weight energy(const V& v, const H& h) const {
        etl::fast_dyn_matrix<weight, V_CV_CHANNELS, K, NH1, NH2> v_cv;      //Temporary convolution

        if(desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY){
            //Definition according to Honglak Lee
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - c sum_v v

            base_type::template compute_vcv<this_type>(v, v_cv, w);

            return - etl::sum(c >> etl::sum_r(v)) - etl::sum(b >> etl::sum_r(h)) - etl::sum(h >> v_cv(1));
        } else if(desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY){
            //Definition according to Honglak Lee / Mixed with Gaussian
            //E(v,h) = - sum_k hk . (Wk*v) - sum_k bk sum_h hk - sum_v ((v - c) ^ 2 / 2)

            base_type::template compute_vcv<this_type>(v, v_cv, w);

            return -sum(etl::pow(v - etl::rep<NV1, NV2>(c), 2) / 2.0) - etl::sum(b >> etl::sum_r(h)) - etl::sum(h >> v_cv(1));
        } else {
            return 0.0;
        }
    }

    template<typename V, typename H, cpp_disable_if(etl::is_etl_expr<V>::value)>
    weight energy(const V& v, const H& h) const {
        etl::fast_dyn_matrix<weight, NC, NV1, NV2> ev;
        etl::fast_dyn_matrix<weight, K, NH1, NH2> eh;

        ev = v;
        eh = h;

        return energy(ev, eh);
    }

    template<typename V>
    weight free_energy_impl(const V& v) const {
        //TODO This function takes ages to compile, must be improved
        //     At least 5 seconds to be compiled on GCC-4.9

        etl::fast_dyn_matrix<weight, V_CV_CHANNELS, K, NH1, NH2> v_cv;      //Temporary convolution

        if(desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY){
            //Definition computed from E(v,h)

            base_type::template compute_vcv<this_type>(v, v_cv, w);

            auto x = etl::rep<NH1, NH2>(b) + v_cv(1);

            return - etl::sum(c >> etl::sum_r(v)) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else if(desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY){
            //Definition computed from E(v,h)

            base_type::template compute_vcv<this_type>(v, v_cv, w);

            auto x = etl::rep<NH1, NH2>(b) + v_cv(1);

            return -sum(etl::pow(v - etl::rep<NV1, NV2>(c), 2) / 2.0) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else {
            return 0.0;
        }
    }

    template<typename V>
    weight free_energy(const V& v) const {
        etl::fast_dyn_matrix<weight, NC, NV1, NV2> ev;
        ev = v;
        return free_energy_impl(ev);
    }

    weight free_energy() const {
        return free_energy_impl(v1);
    }

    //Utilities for DBNs

    //TODO These should really all be renamed...
    using input_deep_t = etl::fast_dyn_matrix<weight, NC, NV1, NV2>;
    using output_deep_t= etl::fast_dyn_matrix<weight, K, NH1, NH2>;
    using input_one_t = etl::fast_dyn_matrix<weight, NC, NV1, NV2>;
    using output_one_t = etl::fast_dyn_matrix<weight, K, NH1, NH2>;
    using input_t = std::vector<input_one_t>;
    using output_t = std::vector<output_one_t>;

    template<typename Input>
    static output_t prepare_output(std::size_t samples){
        return output_t{samples};
    }

    template<typename Input>
    static output_one_t prepare_one_output(){
        return output_one_t{};
    }

    void activate_hidden(output_one_t& h_a, const input_one_t& input) const {
        activate_hidden<true,false>(h_a, h_a, input, input);
    }

    template<typename V, typename H>
    void batch_activate_hidden(H& h_a, const V& input) const {
        etl::fast_dyn_matrix<weight, etl::decay_traits<H>::template dim<0>(), V_CV_CHANNELS, K, NH1, NH2> v_cv;      //Temporary convolution
        batch_activate_hidden<true,false>(h_a, h_a, input, input, v_cv);
    }

    void activate_many(const input_t& input, output_t& h_a, output_t& h_s) const {
        for(std::size_t i = 0; i < input.size(); ++i){
            activate_one(input[i], h_a[i], h_s[i]);
        }
    }

    void activate_many(const input_t& input, output_t& h_a) const {
        for(std::size_t i = 0; i < input.size(); ++i){
            activate_one(input[i], h_a[i]);
        }
    }

private:
    template<typename V1, typename V2, std::size_t Off = 0>
    static void validate_inputs(){
        static_assert(etl::decay_traits<V1>::dimensions() == 3 + Off, "Inputs must be 3D");
        static_assert(etl::decay_traits<V2>::dimensions() == 3 + Off, "Inputs must be 3D");

        static_assert(etl::decay_traits<V1>::template dim<0 + Off>() == NC, "Invalid number of input channels");
        static_assert(etl::decay_traits<V1>::template dim<1 + Off>() == NV1, "Invalid input dimensions");
        static_assert(etl::decay_traits<V1>::template dim<2 + Off>() == NV2, "Invalid input dimensions");

        static_assert(etl::decay_traits<V2>::template dim<0 + Off>() == NC, "Invalid number of input channels");
        static_assert(etl::decay_traits<V2>::template dim<1 + Off>() == NV1, "Invalid input dimensions");
        static_assert(etl::decay_traits<V2>::template dim<2 + Off>() == NV2, "Invalid input dimensions");
    }

    template<typename H1, typename H2, std::size_t Off = 0>
    static void validate_outputs(){
        static_assert(etl::decay_traits<H1>::dimensions() == 3 + Off, "Outputs must be 3D");
        static_assert(etl::decay_traits<H2>::dimensions() == 3 + Off, "Outputs must be 3D");

        static_assert(etl::decay_traits<H1>::template dim<0 + Off>() == K, "Invalid number of output channels");
        static_assert(etl::decay_traits<H1>::template dim<1 + Off>() == NH1, "Invalid output dimensions");
        static_assert(etl::decay_traits<H1>::template dim<2 + Off>() == NH2, "Invalid output dimensions");

        static_assert(etl::decay_traits<H2>::template dim<0 + Off>() == K, "Invalid number of output channels");
        static_assert(etl::decay_traits<H2>::template dim<1 + Off>() == NH1, "Invalid output dimensions");
        static_assert(etl::decay_traits<H2>::template dim<2 + Off>() == NH2, "Invalid output dimensions");
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
