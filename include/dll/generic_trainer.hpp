//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_GENERIC_TRAINER_HPP
#define DBN_GENERIC_TRAINER_HPP

#include "decay_type.hpp"
#include "stop_watch.hpp"
#include "utils.hpp"
#include "batch.hpp"

namespace dll {

template<typename RBM>
struct generic_trainer {
    using rbm_t = RBM;

    template<typename R>
    using trainer_t = typename rbm_t::template trainer_t<R>;

    void train(RBM& rbm, const std::vector<vector<typename RBM::weight>>& training_data, std::size_t max_epochs) const {
        stop_watch<std::chrono::seconds> watch;

        std::cout << "RBM: Train with learning_rate=" << rbm.learning_rate;

        if(rbm_t::Momentum){
            std::cout << ", momentum=" << rbm.momentum;
        }

        if(rbm_t::Decay != DecayType::NONE){
            std::cout << ", weight_cost=" << rbm.weight_cost;
        }

        if(rbm_t::Sparsity){
            std::cout << ", sparsity_target=" << rbm.sparsity_target;
        }

        std::cout << std::endl;

        //TODO Probably shouldn't be here
        if(rbm_t::Init){
            //Initialize the visible biases to log(pi/(1-pi))
            for(size_t i = 0; i < rbm.num_visible; ++i){
                size_t c = 0;
                for(auto& items : training_data){
                    if(items[i] == 1){
                        ++c;
                    }
                }

                auto pi = static_cast<typename rbm_t::weight>(c) / training_data.size();
                pi += 0.0001;
                rbm.a(i) = log(pi / (1.0 - pi));

                dll_assert(std::isfinite(a(i)), "NaN verify");
            }
        }

        auto trainer = make_unique<trainer_t<rbm_t>>();

        auto batches = training_data.size() / rbm_t::BatchSize + (training_data.size() % rbm_t::BatchSize == 0 ? 0 : 1);

        for(size_t epoch= 0; epoch < max_epochs; ++epoch){
            typename rbm_t::weight error = 0.0;

            for(size_t i = 0; i < batches; ++i){
                auto start = i * rbm_t::BatchSize;
                auto end = std::min(start + rbm_t::BatchSize, training_data.size());

                dll::batch<vector<typename rbm_t::weight>> batch(training_data.begin() + start, training_data.begin() + end);
                error += trainer->train_batch(batch, rbm);
            }

            printf("epoch %ld - Reconstruction error average: %.3f - Free energy: %.3f\n",
                epoch, error / batches, rbm.free_energy());

            if(rbm_t::Momentum && epoch == 6){
                rbm.momentum = 0.9;
            }

            //TODO Move that elsewhere
            if(rbm_t::Debug){
                rbm.generate_hidden_images(epoch);
                rbm.generate_histograms(epoch);
            }
        }

        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;
    }
};

} //end of dbn namespace

#endif