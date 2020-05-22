//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <memory>

#include "cpp_utils/algorithm.hpp"

#include "dll/decay_type.hpp"
#include "dll/util/batch.hpp"
#include "dll/util/timers.hpp"
#include "dll/util/random.hpp"
#include "dll/layer_traits.hpp"
#include "dll/trainer/rbm_trainer_fwd.hpp"
#include "dll/trainer/rbm_training_context.hpp"

namespace dll {

enum class init_watcher_t { INIT };
constexpr init_watcher_t init_watcher = init_watcher_t::INIT;

template <typename RBM, typename RW, typename Enable = void>
struct watcher_type {
    using watcher_t = typename RBM::desc::template watcher_t<RBM>;
};

template <typename RBM, typename RW>
struct watcher_type<RBM, RW, std::enable_if_t<!std::is_void_v<RW>>> {
    using watcher_t = RW;
};

/*!
 * \brief A generic trainer for Restricted Boltzmann Machine
 *
 * This trainer use the specified trainer of the RBM to perform unsupervised
 * training.
 */
template <typename RBM, bool EnableWatcher, typename RW>
struct rbm_trainer {
    template <typename R>
    using trainer_t = typename RBM::desc::template trainer_t<R>;

    using rbm_t        = RBM;                                         ///< The RBM type being trained
    using error_type   = typename rbm_t::weight;                      ///< The error data type
    using trainer_type = std::unique_ptr<trainer_t<rbm_t>>;           ///< The type of the trainer
    using watcher_t    = typename watcher_type<rbm_t, RW>::watcher_t; ///< The type of the watcher

    mutable watcher_t watcher; ///< The watcher

    /*!
     * \brief construct a new rbm_trainer, default-initializing the watcher
     */
    rbm_trainer()
            : watcher() {}

    /*!
     * \brief construct a new rbm_trainer, initializing the watcher with the
     * given arguments.
     */
    template <typename... Arg>
    rbm_trainer(init_watcher_t /*init*/, Arg... args)
            : watcher(args...) {}

    static constexpr size_t batch_size = RBM::batch_size; ///< The batch size for pretraining

    size_t total_batches  = 0;   ///< The total number of batches
    error_type last_error = 0.0; ///< The last training error

    //Note: input_first/input_last only relevant for its size, not
    //values since they can point to the input of the first level
    //and not the current level
    template <typename Generator>
    void init_training(RBM& rbm, Generator& generator) {
        rbm.momentum = rbm.initial_momentum;

        if (EnableWatcher) {
            watcher.training_begin(rbm);
        }

        auto size = generator.size();

        //TODO Better handling of incomplete batch size would solve this problem (this could be done by
        //cleaning the data before the last batch)
        if (size % batch_size != 0) {
#ifndef DLL_SILENT
            std::cout << "WARNING: The number of samples should be divisible by the batch size" << std::endl;
            std::cout << "         This may cause discrepancies in the results." << std::endl;
#endif
        }

        //Only used for debugging purposes, no need to be precise
        total_batches = size / batch_size;

        last_error = 0.0;
    }

    /*!
     * \brief Return the trainer for the given RBM
     */
    static trainer_type get_trainer(RBM& rbm) {
        //Allocate the trainer on the heap (may be large)
        return std::make_unique<trainer_t<rbm_t>>(rbm);
    }

    error_type finalize_training(RBM& rbm) {
        if (EnableWatcher) {
            watcher.training_end(rbm);
        }

        return last_error;
    }

    template <typename Generator>
    error_type train(RBM& rbm, Generator & generator, size_t max_epochs) {
        dll::auto_timer timer("rbm_trainer:train");

        //Initialize RBM and trainign parameters
        init_training(rbm, generator);

        //Some RBM may init weights based on the training data
        //Note: This can't be done in init_training, since it will
        //sometimes be called with the wrong input values
        if constexpr (rbm_layer_traits<rbm_t>::init_weights()){
            rbm.init_weights(generator);
        }

        //Allocate the trainer
        auto trainer = get_trainer(rbm);

        //Train for max_epochs epoch
        for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
            //Shuffle if necessary
            if(rbm_layer_traits<rbm_t>::has_shuffle()){
                generator.reset_shuffle();
            } else {
                generator.reset();
            }

            // Set the the generator in train mode
            generator.set_train();

            //Create a new context for this epoch
            rbm_training_context context;

            //Start a new epoch
            init_epoch(epoch);

            //Train on all the data
            train_sub(generator, trainer, context, rbm);

            //Finalize the current epoch
            finalize_epoch(epoch, context, rbm);
        }

        return finalize_training(rbm);
    }

    size_t batches = 0; ///< The number of batches
    size_t samples = 0; ///< The number of samples

    /*!
     * \brief Initialization of the epoch
     */
    void init_epoch(size_t epoch) {
        batches = 0;
        samples = 0;

        //Notify the watcher
        if (EnableWatcher) {
            watcher.epoch_start(epoch);
        }
    }

    template <typename Generator>
    void train_sub(Generator& generator, trainer_type& trainer, rbm_training_context& context, rbm_t& rbm) {
        while(generator.has_next_batch()){
            //Train the batch
            train_batch(generator.data_batch(), generator.label_batch(), trainer, context, rbm);

            // Go to the next batch
            generator.next_batch();
        }
    }

    template <typename InputBatch, typename ExpectedBatch>
    void train_batch(InputBatch&& input, ExpectedBatch&& expected, trainer_type& trainer, rbm_training_context& context, rbm_t& rbm) {
        ++batches;

        trainer->train_batch(input, expected, context);

        context.reconstruction_error += context.batch_error;
        context.sparsity += context.batch_sparsity;

        if constexpr (EnableWatcher && rbm_layer_traits<rbm_t>::free_energy()) {
            for (auto& v : input) {
                context.free_energy += rbm.free_energy(v);
            }
        }

        if (EnableWatcher && rbm_layer_traits<rbm_t>::is_verbose()) {
            watcher.batch_end(rbm, context, batches, total_batches);
        }
    }

    void finalize_epoch(size_t epoch, rbm_training_context& context, rbm_t& rbm) {
        //Average all the gathered information
        context.reconstruction_error /= batches;
        context.sparsity /= batches;
        context.free_energy /= samples;

        //After some time increase the momentum
        if (rbm_layer_traits<rbm_t>::has_momentum() && epoch == rbm.final_momentum_epoch) {
            rbm.momentum = rbm.final_momentum;
        }

        //Notify the watcher
        if (EnableWatcher) {
            watcher.epoch_end(epoch, context, rbm);
        }

        //Save the error for the return value
        last_error = context.reconstruction_error;
    }
};

} //end of dll namespace
