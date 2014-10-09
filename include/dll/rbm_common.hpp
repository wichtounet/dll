//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_RBM_COMMON_HPP
#define DLL_RBM_COMMON_HPP

#include "rbm_traits.hpp"

namespace dll {

namespace rbm_detail {

template<typename RBM>
static void store(std::ostream& os, const RBM& rbm){
    binary_write_all(os, rbm.w);
    binary_write_all(os, rbm.b);
    binary_write_all(os, rbm.c);
}

template<typename RBM>
static void load(std::istream& is, RBM& rbm){
    binary_load_all(is, rbm.w);
    binary_load_all(is, rbm.b);
    binary_load_all(is, rbm.c);
}

template<typename Iterator, typename RBM>
void init_weights(Iterator first, Iterator last, RBM& rbm){
    auto size = std::distance(first, last);

    //Initialize the visible biases to log(pi/(1-pi))
    for(size_t i = 0; i < num_visible(rbm); ++i){
        auto count = std::count_if(first, last,
            [i](auto& a){return a[i] == 1; });

        auto pi = static_cast<double>(count) / size;
        pi += 0.0001;
        rbm.c(i) = log(pi / (1.0 - pi));

        cpp_assert(std::isfinite(rbm.c(i)), "NaN verify");
    }
}

template<typename RBM, typename V, typename H, cpp::enable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
typename RBM::weight energy(const RBM& rbm, const V& v, const H& h){
    if(RBM::desc::visible_unit == unit_type::BINARY && RBM::desc::hidden_unit == unit_type::BINARY){
        //Definition according to G. Hinton
        //E(v,h) = -sum(ai*vi) - sum(bj*hj) -sum(vi*hj*wij)

        auto mid_term = 0.0;

        //TODO Simplify that
        for(size_t i = 0; i < num_visible(rbm); ++i){
            for(size_t j = 0; j < num_hidden(rbm); ++j){
                mid_term += rbm.w(i, j) * v(i) * h(j);
            }
        }

        return -etl::dot(rbm.c, v) - etl::dot(rbm.b, h) - mid_term;
    } else {
        return 0.0;
    }
}

template<typename RBM, typename V, typename H, cpp::disable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
typename RBM::weight energy(const RBM& rbm, const V& v, const H& h){
    if(RBM::desc::visible_unit == unit_type::BINARY && RBM::desc::hidden_unit == unit_type::BINARY){
        //Definition according to G. Hinton
        //E(v,h) = -sum(ai*vi) - sum(bj*hj) -sum(vi*hj*wij)

        auto visible_term = 0.0;
        auto hidden_term = 0.0;

        for(size_t i = 0; i < num_visible(rbm); ++i){
            visible_term = rbm.c(i) * v[i];
        }

        for(size_t j = 0; j < num_hidden(rbm); ++j){
            hidden_term = rbm.b(j) * h[j];
        }

        auto mid_term = 0.0;

        //TODO Simplify that
        for(size_t i = 0; i < num_visible(rbm); ++i){
            for(size_t j = 0; j < num_hidden(rbm); ++j){
                mid_term += rbm.w(i, j) * v(i) * h(j);
            }
        }

        return -visible_term - hidden_term - mid_term;
    } else {
        return 0.0;
    }
}

template<typename RBM, typename V, cpp::enable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
typename RBM::weight free_energy(const RBM& rbm, const V& v){
    if(RBM::desc::visible_unit == unit_type::BINARY && RBM::desc::hidden_unit == unit_type::BINARY){
        //Definition according to G. Hinton
        //F(v) = -sum(ai*vi) - sum(log(1 + e^(xj)))
        //xj = input to hidden neuron j

        using namespace etl;

        static fast_matrix<typename RBM::weight, 1, RBM::num_hidden> t;

        auto x = rbm.b + mmul(reshape<1, RBM::desc::num_visible>(v), rbm.w, t);

        return -dot(rbm.c, v) - sum(log(1.0 + exp(x)));
    } else {
        return 0.0;
    }
}

template<typename RBM, typename V, cpp::disable_if_u<etl::is_etl_expr<V>::value> = cpp::detail::dummy>
typename RBM::weight free_energy(const RBM& rbm, const V& v){
    if(RBM::desc::visible_unit == unit_type::BINARY && RBM::desc::hidden_unit == unit_type::BINARY){
        //Definition according to G. Hinton
        //F(v) = -sum(ai*vi) - sum(log(1 + e^(xj)))
        //xj = input to hidden neuron j

        auto visible_term = 0.0;

        for(size_t i = 0; i < num_visible(rbm); ++i){
            visible_term = rbm.c(i) * v[i];
        }

        auto mid_term = 0.0;

        for(size_t j = 0; j < num_hidden(rbm); ++j){
            auto x = rbm.b(j);
            for(size_t i = 0; i < num_visible(rbm); ++i){
                x += v[i] * rbm.w(i,j);
            }

            mid_term += std::log(1.0 + std::exp(x));
        }

        return -visible_term - mid_term;
    } else {
        return 0.0;
    }
}

template<typename Sample, typename RBM>
void reconstruct(const Sample& items, RBM& rbm){
    cpp_assert(items.size() == num_visible, "The size of the training sample must match visible units");

    cpp::stop_watch<> watch;

    //Set the state of the visible units
    rbm.v1 = items;

    rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);
    rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);
    rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);

    std::cout << "Reconstruction took " << watch.elapsed() << "ms" << std::endl;
}

template<typename RBM>
void display_weights(RBM& rbm){
    for(size_t j = 0; j < num_hidden(rbm); ++j){
        for(size_t i = 0; i < num_visible(rbm); ++i){
            std::cout << rbm.w(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

template<typename RBM>
void display_weights(RBM& rbm, size_t matrix){
    for(size_t j = 0; j < num_hidden(rbm); ++j){
        for(size_t i = 0; i < num_visible(rbm);){
            for(size_t m = 0; m < matrix; ++m){
                std::cout << rbm.w(i++, j) << " ";
            }
            std::cout << std::endl;
        }
    }
}

template<typename RBM>
void display_visible_units(RBM& rbm){
    std::cout << "Visible  Value" << std::endl;

    for(size_t i = 0; i < num_visible(rbm); ++i){
        printf("%-8lu %d\n", i, rbm.v2_s(i));
    }
}

template<typename RBM>
void display_visible_units(RBM& rbm, size_t matrix){
    for(size_t i = 0; i < matrix; ++i){
        for(size_t j = 0; j < matrix; ++j){
            std::cout << rbm.v2_s(i * matrix + j) << " ";
        }
        std::cout << std::endl;
    }
}

template<typename RBM>
void display_hidden_units(RBM& rbm){
    std::cout << "Hidden Value" << std::endl;

    for(size_t j = 0; j < num_hidden(rbm); ++j){
        printf("%-8lu %d\n", j, rbm.h2_s(j));
    }
}

} //end of rbm_detail

} //end of dbn namespace

#endif