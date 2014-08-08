//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_RBM_COMMON_HPP
#define DLL_RBM_COMMON_HPP

namespace dll {

template<typename RBM>
struct rbm_trainer;

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

template<typename Samples, typename RBM>
void init_weights(const Samples& training_data, RBM& rbm){
    //Initialize the visible biases to log(pi/(1-pi))
    for(size_t i = 0; i < num_visible(rbm); ++i){
        auto count = std::count_if(training_data.begin(), training_data.end(),
            [i](auto& a){return a[i] == 1; });

        auto pi = static_cast<double>(count) / training_data.size();
        pi += 0.0001;
        rbm.c(i) = log(pi / (1.0 - pi));

        dll_assert(std::isfinite(rbm.c(i)), "NaN verify");
    }
}

template<typename RBM>
typename RBM::weight free_energy(RBM& rbm){
    typename RBM::weight energy = 0.0;

    for(size_t i = 0; i < num_visible(rbm); ++i){
        for(size_t j = 0; j < num_hidden(rbm); ++j){
            energy += rbm.w(i, j) * rbm.b(j) * rbm.c(i);
        }
    }

    return -energy;
}

template<typename Sample, typename RBM>
void reconstruct(const Sample& items, RBM& rbm){
    dll_assert(items.size() == num_visible, "The size of the training sample must match visible units");

    stop_watch<> watch;

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