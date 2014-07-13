//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_WATCHER_HPP
#define DBN_WATCHER_HPP

#include "stop_watch.hpp"
#include "rbm_traits.hpp"

namespace dll {

template<typename RBM>
struct default_watcher {
    stop_watch<std::chrono::seconds> watch;

    void training_begin(const RBM& rbm){
        std::cout << "RBM: Train with learning_rate=" << rbm.learning_rate;

        if(rbm_traits<RBM>::has_momentum()){
            std::cout << ", momentum=" << rbm.momentum;
        }

        if(rbm_traits<RBM>::decay() != decay_type::NONE){
            std::cout << ", weight_cost=" << rbm.weight_cost;
        }

        if(rbm_traits<RBM>::has_sparsity()){
            std::cout << ", sparsity_target=" << rbm.sparsity_target;
        }

        std::cout << std::endl;
    }

    void epoch_end(std::size_t epoch, double error, const RBM& rbm){
        printf("epoch %ld - Reconstruction error average: %.3f - Free energy: %.3f\n",
            epoch, error, rbm.free_energy());
    }

    void training_end(const RBM&){
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;
    }
};

} //end of dbn namespace

#endif