//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_RBM_TRAINER_HPP
#define DLL_RBM_TRAINER_HPP

#include "decay_type.hpp"
#include "utils.hpp"
#include "batch.hpp"
#include "rbm_traits.hpp"

namespace dll {

/*!
 * \brief A generic trainer for Restricted Boltzmann Machine
 *
 * This trainer use the specified trainer of the RBM to perform unsupervised
 * training.
 */
template<typename RBM>
struct rbm_trainer {
    using rbm_t = RBM;

    template<typename R>
    using trainer_t = typename rbm_t::desc::template trainer_t<R>;

    template<typename R>
    using watcher_t = typename rbm_t::desc::template watcher_t<R>;

    template<typename Samples, typename R = RBM, enable_if_u<rbm_traits<R>::init_weights()> = ::detail::dummy>
    static void init_weights(RBM& rbm, const Samples& training_data){
        rbm.init_weights(training_data);
    }

    template<typename Samples, typename R = RBM, disable_if_u<rbm_traits<R>::init_weights()> = ::detail::dummy>
    static void init_weights(RBM&, const Samples&){
        //NOP
    }

    template<typename Samples>
    typename rbm_t::weight train(RBM& rbm, const Samples& training_data, std::size_t max_epochs) const {
        watcher_t<rbm_t> watcher;

        rbm.momentum = rbm.initial_momentum;

        watcher.training_begin(rbm);

        //Some RBM may init weights based on the training data
        init_weights(rbm, training_data);

        auto trainer = make_unique<trainer_t<rbm_t>>(rbm);

        //Compute the number of batches
        auto batch_size = get_batch_size(rbm);

        auto batches = training_data.size() / batch_size + (training_data.size() % batch_size == 0 ? 0 : 1);

        typename rbm_t::weight last_error = 0.0;

        //Train for max_epochs epoch
        for(size_t epoch= 0; epoch < max_epochs; ++epoch){
            typename rbm_t::weight error = 0.0;

            //Train one mini-batch at a time
            for(size_t i = 0; i < batches; ++i){
                auto start = i * batch_size;
                auto end = std::min(start + batch_size, training_data.size());

                dll::batch<Samples> batch(training_data.begin() + start, training_data.begin() + end);
                error += trainer->train_batch(batch);
            }

            last_error = error / batches;

            //After some time increase the momentum
            if(rbm_traits<rbm_t>::has_momentum() && epoch == rbm.final_momentum_epoch){
                rbm.momentum = rbm.final_momentum;
            }

            watcher.epoch_end(epoch, last_error, rbm);
        }

        watcher.training_end(rbm);

        return last_error;
    }
};

} //end of dbn namespace

#endif