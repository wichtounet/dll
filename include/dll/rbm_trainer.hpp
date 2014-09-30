//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_RBM_TRAINER_HPP
#define DLL_RBM_TRAINER_HPP

#include "decay_type.hpp"
#include "batch.hpp"
#include "rbm_traits.hpp"

namespace dll {

enum class init_watcher_t { INIT };
constexpr const init_watcher_t init_watcher = init_watcher_t::INIT;

template<typename RBM, typename RW, typename Enable = void>
struct watcher_type {
    using watcher_t = typename RBM::desc::template watcher_t<RBM>;
};

template<typename RBM, typename RW>
struct watcher_type<RBM, RW, std::enable_if_t<not_u<std::is_void<RW>::value>::value>> {
    using watcher_t = RW;
};

/*!
 * \brief A generic trainer for Restricted Boltzmann Machine
 *
 * This trainer use the specified trainer of the RBM to perform unsupervised
 * training.
 */
template<typename RBM, bool EnableWatcher, typename RW>
struct rbm_trainer {
    using rbm_t = RBM;

    template<typename R>
    using trainer_t = typename rbm_t::desc::template trainer_t<R>;

    using watcher_t = typename watcher_type<rbm_t, RW>::watcher_t;

    mutable watcher_t watcher;

    rbm_trainer() : watcher() {}

    template<typename... Arg>
    rbm_trainer(init_watcher_t /*init*/, Arg... args) : watcher(args...) {}

    template<typename Iterator, typename R = RBM, enable_if_u<rbm_traits<R>::init_weights()> = ::detail::dummy>
    static void init_weights(RBM& rbm, Iterator first, Iterator last){
        rbm.init_weights(first, last);
    }

    template<typename Iterator, typename R = RBM, disable_if_u<rbm_traits<R>::init_weights()> = ::detail::dummy>
    static void init_weights(RBM&, Iterator, Iterator){
        //NOP
    }

    //TODO This should be removed later
    template<typename Samples>
    typename rbm_t::weight train(RBM& rbm, const Samples& training_data, std::size_t max_epochs) const {
        return train(rbm, training_data.begin(), training_data.end(), max_epochs);
    }

    template<typename Iterator>
    typename rbm_t::weight train(RBM& rbm, Iterator first, Iterator last, std::size_t max_epochs) const {
        rbm.momentum = rbm.initial_momentum;

        if(EnableWatcher){
            watcher.training_begin(rbm);
        }

        //Some RBM may init weights based on the training data
        init_weights(rbm, first, last);

        auto trainer = std::make_unique<trainer_t<rbm_t>>(rbm);

        //Compute the number of batches
        auto batch_size = get_batch_size(rbm);

        typename rbm_t::weight last_error = 0.0;

        //Train for max_epochs epoch
        for(size_t epoch= 0; epoch < max_epochs; ++epoch){
            typename rbm_t::weight error = 0.0;
            typename rbm_t::weight free_energy = 0.0;

            auto it = first;
            auto end = last;

            std::size_t batches = 0;
            std::size_t samples = 0;

            while(it != end){
                auto start = it;

                std::size_t i = 0;
                while(it != end && i < batch_size){
                    ++it;
                    ++samples;
                    ++i;
                }

                ++batches;

                auto batch = make_batch(start, it);
                error += trainer->train_batch(batch);

                if(EnableWatcher){
                    for(auto& v : batch){
                        free_energy += rbm.free_energy(v);
                    }
                }
            }

            last_error = error / batches;
            free_energy /= samples;

            //After some time increase the momentum
            if(rbm_traits<rbm_t>::has_momentum() && epoch == rbm.final_momentum_epoch){
                rbm.momentum = rbm.final_momentum;
            }

            if(EnableWatcher){
                watcher.epoch_end(epoch, last_error, free_energy, rbm);
            }
        }

        if(EnableWatcher){
            watcher.training_end(rbm);
        }

        return last_error;
    }
};

} //end of dbn namespace

#endif