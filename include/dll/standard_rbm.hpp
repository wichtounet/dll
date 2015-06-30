//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_STANDARD_RBM_HPP
#define DLL_STANDARD_RBM_HPP

#include <cmath>
#include <vector>
#include <random>
#include <functional>
#include <ctime>

#include "cpp_utils/stop_watch.hpp"    //Performance counter
#include "cpp_utils/assert.hpp"

#include "etl/etl.hpp"

#include "rbm_base.hpp"         //The base class
#include "base_conf.hpp"
#include "math.hpp"
#include "io.hpp"
#include "checks.hpp"           //NaN checks

namespace dll {

/*!
 * \brief Standard version of Restricted Boltzmann Machine
 *
 * This follows the definition of a RBM by Geoffrey Hinton.
 */
template<typename Parent, typename Desc>
struct standard_rbm : public rbm_base<Parent, Desc> {
    using desc = Desc;
    using parent_t = Parent;
    using this_type = standard_rbm<parent_t, desc>;
    using base_type = rbm_base<parent_t, desc>;
    using weight = typename desc::weight;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static_assert(visible_unit != unit_type::SOFTMAX, "Softmax Visible units are not support");
    static_assert(hidden_unit != unit_type::GAUSSIAN, "Gaussian hidden units are not supported");

    standard_rbm(){
        //Better initialization of learning rate
        base_type::learning_rate =
                visible_unit == unit_type::GAUSSIAN && is_relu(hidden_unit) ? 1e-5
            :   visible_unit == unit_type::GAUSSIAN || is_relu(hidden_unit) ? 1e-3
            :   /* Only ReLU and Gaussian Units needs lower rate */           1e-1;
    }

    //Energy functions

    template<typename V, typename H>
    weight energy(const V& v, const H& h) const {
        return free_energy(*static_cast<const parent_t*>(this), v, h);
    }

    template<typename V>
    weight free_energy(const V& v) const {
        return free_energy(*static_cast<const parent_t*>(this), v);
    }

    weight free_energy() const {
        auto& rbm = *static_cast<const parent_t*>(this);
        return free_energy(rbm, rbm.v1);
    }

    //Various functions

    template<typename Iterator>
    void init_weights(Iterator&& first, Iterator&& last){
        init_weights(std::forward<Iterator>(first), std::forward<Iterator>(last), *static_cast<parent_t*>(this));
    }

    template<typename Sample>
    void reconstruct(const Sample& items){
        reconstruct(items, *static_cast<parent_t*>(this));
    }

    //Display functions

    void display_units() const {
        display_visible_units();
        display_hidden_units();
    }

    void display_visible_units() const {
        display_visible_units(*static_cast<const parent_t*>(this));
    }

    void display_visible_units(std::size_t matrix) const {
        display_visible_units(*static_cast<const parent_t*>(this), matrix);
    }

    void display_hidden_units() const {
        display_hidden_units(*static_cast<const parent_t*>(this));
    }

    void display_weights() const {
        display_weights(*static_cast<const parent_t*>(this));
    }

    void display_weights(std::size_t matrix) const {
        display_weights(matrix, *static_cast<const parent_t*>(this));
    }

protected:

    //Since the sub classes does not have the same fields, it is not possible
    //to put the fields in standard_rbm, therefore, it is necessary to use template
    //functions to implement the details

    template<typename Iterator, typename RBM>
    static void init_weights(Iterator first, Iterator last, RBM& rbm){
        auto size = std::distance(first, last);

        //Initialize the visible biases to log(pi/(1-pi))
        for(std::size_t i = 0; i < num_visible(rbm); ++i){
            auto count = std::count_if(first, last,
                [i](auto& a){return a[i] == 1; });

            auto pi = static_cast<double>(count) / size;
            pi += 0.0001;
            rbm.c(i) = log(pi / (1.0 - pi));

            cpp_assert(std::isfinite(rbm.c(i)), "NaN verify");
        }
    }

    template<typename Sample, typename RBM>
    static void reconstruct(const Sample& items, RBM& rbm){
        cpp_assert(items.size() == num_visible(rbm), "The size of the training sample must match visible units");

        cpp::stop_watch<> watch;

        //Set the state of the visible units
        rbm.v1 = items;

        rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);
        rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);
        rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);

        std::cout << "Reconstruction took " << watch.elapsed() << "ms" << std::endl;
    }

    template<typename RBM>
    static void display_weights(RBM& rbm){
        for(std::size_t j = 0; j < num_hidden(rbm); ++j){
            for(std::size_t i = 0; i < num_visible(rbm); ++i){
                std::cout << rbm.w(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    template<typename RBM>
    static void display_weights(RBM& rbm, std::size_t matrix){
        for(std::size_t j = 0; j < num_hidden(rbm); ++j){
            for(std::size_t i = 0; i < num_visible(rbm);){
                for(std::size_t m = 0; m < matrix; ++m){
                    std::cout << rbm.w(i++, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    template<typename RBM>
    static void display_visible_units(RBM& rbm){
        std::cout << "Visible  Value" << std::endl;

        for(std::size_t i = 0; i < num_visible(rbm); ++i){
            printf("%-8lu %d\n", i, rbm.v2_s(i));
        }
    }

    template<typename RBM>
    static void display_visible_units(RBM& rbm, std::size_t matrix){
        for(std::size_t i = 0; i < matrix; ++i){
            for(std::size_t j = 0; j < matrix; ++j){
                std::cout << rbm.v2_s(i * matrix + j) << " ";
            }
            std::cout << std::endl;
        }
    }

    template<typename RBM>
    static void display_hidden_units(RBM& rbm){
        std::cout << "Hidden Value" << std::endl;

        for(std::size_t j = 0; j < num_hidden(rbm); ++j){
            printf("%-8lu %d\n", j, rbm.h2_s(j));
        }
    }

    //Note: Considering that energy and free energy are not critical, their implementations
    //are not highly optimized.

    template<typename RBM, typename V, typename H, cpp::enable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
    static typename RBM::weight energy(const RBM& rbm, const V& v, const H& h){
        if(RBM::desc::visible_unit == unit_type::BINARY && RBM::desc::hidden_unit == unit_type::BINARY){
            //Definition according to G. Hinton
            //E(v,h) = -sum(ai*vi) - sum(bj*hj) -sum(vi*hj*wij)

            auto x = rbm.b + v * rbm.w;

            return -etl::dot(rbm.c, v) - etl::dot(rbm.b, h) - etl::sum(x);
        } else if(RBM::desc::visible_unit == unit_type::GAUSSIAN && RBM::desc::hidden_unit == unit_type::BINARY){
            //Definition according to G. Hinton
            //E(v,h) = -sum((vi - ai)^2/(2*var*var)) - sum(bj*hj) -sum((vi/var)*hj*wij)

            auto x = rbm.b + v * rbm.w;

            return etl::sum(etl::pow(v - rbm.c, 2) / 2.0) - etl::dot(rbm.b, h) - etl::sum(x);
        } else {
            return 0.0;
        }
    }

    template<typename RBM, typename V, typename H, cpp::disable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
    static typename RBM::weight energy(const RBM& rbm, const V& v, const H& h){
        etl::dyn_vector<typename V::value_type> ev(v);
        etl::dyn_vector<typename H::value_type> eh(h);
        return energy(rbm, ev, eh);
    }

    //Free energy are computed from the E(v,h) formulas
    //1. by isolating hi in the E(v,h) formulas
    //2. by using the sum_hi which sums over all the possible values of hi
    //3. by considering only binary hidden units, the values are only 0 and 1
    //and therefore the values can be "integrated out" easily.

    template<typename RBM, typename V, cpp::enable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
    static typename RBM::weight free_energy(const RBM& rbm, const V& v){
        if(RBM::desc::visible_unit == unit_type::BINARY && RBM::desc::hidden_unit == unit_type::BINARY){
            //Definition according to G. Hinton
            //F(v) = -sum(ai*vi) - sum(log(1 + e^(xj)))

            auto x = rbm.b + v * rbm.w;

            return -etl::dot(rbm.c, v) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else if(RBM::desc::visible_unit == unit_type::GAUSSIAN && RBM::desc::hidden_unit == unit_type::BINARY){
            //Definition computed from E(v,h)
            //F(v) = sum((vi-ai)^2/2) - sum(log(1 + e^(xj)))

            auto x = rbm.b + v * rbm.w;

            return etl::sum(etl::pow(v - rbm.c, 2) / 2.0) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else {
            return 0.0;
        }
    }

    template<typename RBM, typename V, cpp::disable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
    static typename RBM::weight free_energy(const RBM& rbm, const V& v){
        etl::dyn_vector<typename V::value_type> ev(v);
        return free_energy(rbm, ev);
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V, typename B, typename W, typename T>
    static void std_activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V&, const B& b, const W& w, T&& t){
        using namespace etl;

        //Compute activation probabilities
        if(P){
            if(hidden_unit == unit_type::BINARY){
                h_a = sigmoid(b + mul(v_a, w, t));
            } else if(hidden_unit == unit_type::RELU){
                h_a = max(b + mul(v_a, w, t), 0.0);
            } else if(hidden_unit == unit_type::RELU6){
                h_a = min(max(b + mul(v_a, w, t), 0.0), 6.0);
            } else if(hidden_unit == unit_type::RELU1){
                h_a = min(max(b + mul(v_a, w, t), 0.0), 1.0);
            } else if(hidden_unit == unit_type::SOFTMAX){
                h_a = stable_softmax(b + mul(v_a, w, t));
            }

            nan_check_deep(h_a);

            //Compute sampled values directly
            if(S){
                if(hidden_unit == unit_type::BINARY){
                    h_s = bernoulli(h_a);
                } else if(hidden_unit == unit_type::RELU){
                    h_s = logistic_noise(h_a); //TODO This is probably wrong
                } else if(hidden_unit == unit_type::RELU6){
                    h_s = ranged_noise(h_a, 6.0); //TODO This is probably wrong
                } else if(hidden_unit == unit_type::RELU1){
                    h_s = ranged_noise(h_a, 1.0); //TODO This is probably wrong
                } else if(hidden_unit == unit_type::SOFTMAX){
                    h_s = one_if_max(h_a);
                }

                nan_check_deep(h_s);
            }
        }
        //Compute sampled values
        else if(S){
            if(hidden_unit == unit_type::BINARY){
                h_s = bernoulli(sigmoid(b + mul(v_a, w, t)));
            } else if(hidden_unit == unit_type::RELU){
                h_s = logistic_noise(max(b + mul(v_a, w, t), 0.0)); //TODO This is probably wrong
            } else if(hidden_unit == unit_type::RELU6){
                h_s = ranged_noise(min(max(b + mul(v_a, w, t), 0.0), 6.0), 6.0); //TODO This is probably wrong
            } else if(hidden_unit == unit_type::RELU1){
                h_s = ranged_noise(min(max(b + mul(v_a, w, t), 0.0), 1.0), 1.0); //TODO This is probably wrong
            } else if(hidden_unit == unit_type::SOFTMAX){
                h_s = one_if_max(stable_softmax(b + mul(v_a, w, t)));
            }

            nan_check_deep(h_s);
        }
    }

    template<bool P = true, bool S = true, typename H, typename V, typename C, typename W, typename T>
    static void std_activate_visible(const H&, const H& h_s, V&& v_a, V&& v_s, const C& c, const W& w, T&& t){
        using namespace etl;

        if(P){
            if(visible_unit == unit_type::BINARY){
                v_a = sigmoid(c + mul(w, h_s, t));
            } else if(visible_unit == unit_type::GAUSSIAN){
                v_a = c + mul(w, h_s, t);
            } else if(visible_unit == unit_type::RELU){
                v_a = max(c + mul(w, h_s, t), 0.0);
            }

            nan_check_deep(v_a);
        }

        if(S){
            if(visible_unit == unit_type::BINARY){
                v_s = bernoulli(sigmoid(c + mul(w, h_s, t)));
            } else if(visible_unit == unit_type::GAUSSIAN){
                v_s = c + mul(w, h_s, t);
            } else if(visible_unit == unit_type::RELU){
                v_s = logistic_noise(max(c + mul(w, h_s, t), 0.0));
            }

            nan_check_deep(v_s);
        }
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V, typename B, typename W>
    static void batch_std_activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V&, const B& b, const W& w){
        using namespace etl;

        const auto Batch = etl::dim<0>(h_a);

        cpp_assert(etl::dim<0>(h_s) == Batch && etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");

        //Compute activation probabilities
        if(P){
            if(hidden_unit == unit_type::BINARY){
                h_a = sigmoid(rep_l(b, Batch) + mul(v_a, w));
            } else if(hidden_unit == unit_type::RELU){
                h_a = max(rep_l(b, Batch) + mul(v_a, w), 0.0);
            } else if(hidden_unit == unit_type::RELU6){
                h_a = min(max(rep_l(b, Batch) + mul(v_a,w), 0.0), 6.0);
            } else if(hidden_unit == unit_type::RELU1){
                h_a = min(max(rep_l(b, Batch) + mul(v_a, w), 0.0), 1.0);
            } else if(hidden_unit == unit_type::SOFTMAX){
                h_a = stable_softmax(rep_l(b, Batch) + mul(v_a, w));
            }

            nan_check_deep(h_a);

            //Compute sampled values directly
            if(S){
                if(hidden_unit == unit_type::BINARY){
                    h_s = bernoulli(h_a);
                } else if(hidden_unit == unit_type::RELU){
                    h_s = logistic_noise(h_a); //TODO This is probably wrong
                } else if(hidden_unit == unit_type::RELU6){
                    h_s = ranged_noise(h_a, 6.0); //TODO This is probably wrong
                } else if(hidden_unit == unit_type::RELU1){
                    h_s = ranged_noise(h_a, 1.0); //TODO This is probably wrong
                } else if(hidden_unit == unit_type::SOFTMAX){
                    h_s = one_if_max(h_a);
                }

                nan_check_deep(h_s);
            }
        }
        //Compute sampled values
        else if(S){
            if(hidden_unit == unit_type::BINARY){
                h_s = bernoulli(sigmoid(rep_l(b, Batch) + mul(v_a, w)));
            } else if(hidden_unit == unit_type::RELU){
                h_s = logistic_noise(max(rep_l(b, Batch) + mul(v_a, w), 0.0)); //TODO This is probably wrong
            } else if(hidden_unit == unit_type::RELU6){
                h_s = ranged_noise(min(max(rep_l(b, Batch) + mul(v_a, w), 0.0), 6.0), 6.0); //TODO This is probably wrong
            } else if(hidden_unit == unit_type::RELU1){
                h_s = ranged_noise(min(max(rep_l(b, Batch) + mul(v_a, w), 0.0), 1.0), 1.0); //TODO This is probably wrong
            } else if(hidden_unit == unit_type::SOFTMAX){
                h_s = one_if_max(stable_softmax(rep_l(b, Batch) + mul(v_a, w)));
            }

            nan_check_deep(h_s);
        }
    }

    template<bool P = true, bool S = true, typename H, typename V, typename C, typename W>
    static void batch_std_activate_visible(const H&, const H& h_s, V&& v_a, V&& v_s, const C& c, const W& w){
        using namespace etl;

        const auto Batch = etl::dim<0>(v_s);

        cpp_assert(etl::dim<0>(h_s) == Batch && etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");

        if(P){
            if(visible_unit == unit_type::BINARY){
                v_a = sigmoid(rep_l(c,Batch) + transpose(mul(w, transpose(h_s))));
            } else if(visible_unit == unit_type::GAUSSIAN){
                v_a = rep_l(c,Batch) + transpose(mul(w, transpose(h_s)));
            } else if(visible_unit == unit_type::RELU){
                v_a = max(rep_l(c,Batch) + transpose(mul(w, transpose(h_s))), 0.0);
            }

            nan_check_deep(v_a);
        }

        if(S){
            if(visible_unit == unit_type::BINARY){
                v_s = bernoulli(sigmoid(rep_l(c,Batch) + transpose(mul(w, transpose(h_s)))));
            } else if(visible_unit == unit_type::GAUSSIAN){
                v_s = rep_l(c,Batch) + transpose(mul(w, transpose(h_s)));
            } else if(visible_unit == unit_type::RELU){
                v_s = logistic_noise(max(rep_l(c,Batch) + transpose(mul(w, transpose(h_s))), 0.0));
            }

            nan_check_deep(v_s);
        }
    }

public:

    //Utilities to be used by DBNs

    using input_one_t = etl::dyn_vector<weight>;
    using output_one_t = etl::dyn_vector<weight>;
    using input_t = std::vector<input_one_t>;
    using output_t = std::vector<output_one_t>;

    template<typename Iterator>
    auto convert_input(const Iterator& first, const Iterator& last) const {
        input_t input;

        if(std::is_same<typename std::iterator_traits<Iterator>::iterator_category, std::random_access_iterator_tag>::value){
            input.reserve(std::distance(first, last));
        }

        auto& derived = *static_cast<const parent_t*>(this);

        std::for_each(first, last, [&input, &derived](auto& sample){
            input.emplace_back(derived.input_size());
            input.back() = sample;
        });

        return input;
    }

    template<typename Sample>
    input_one_t convert_sample(const Sample& sample) const {
        input_one_t input(static_cast<const parent_t*>(this)->input_size());
        input = sample;
        return input;
    }

    template<typename Input>
    output_t prepare_output(std::size_t samples, bool is_last = false, std::size_t labels = 0) const {
        output_t output;
        output.reserve(samples);

        for(std::size_t i = 0; i < samples; ++i){
            output.emplace_back(static_cast<const parent_t*>(this)->output_size() + (is_last ? labels : 0));
        }

        return output;
    }

    template<typename Input>
    output_one_t prepare_one_output(bool is_last = false, std::size_t labels = 0) const {
        return output_one_t(static_cast<const parent_t*>(this)->output_size() + (is_last ? labels : 0));
    }

    input_one_t prepare_one_input() const {
        return input_one_t(static_cast<const parent_t*>(this)->input_size());
    }

    void activate_one(const input_one_t& input, output_one_t& h_a, output_one_t& h_s) const {
        static_cast<const parent_t*>(this)->activate_hidden(h_a, h_s, input, input);
    }

    void activate_one(const input_one_t& input, output_one_t& h_a) const {
        static_cast<const parent_t*>(this)->template activate_hidden<true, false>(h_a, h_a, input, input);
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
};

} //end of dll namespace

#endif
