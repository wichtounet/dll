//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/algorithm.hpp" // For parallel_shuffle

#include "etl/etl.hpp"

#include "dll/util/labels.hpp"
#include "dll/util/timers.hpp"
#include "dll/util/random.hpp"
#include "dll/util/batch.hpp" // For make_batch
#include "dll/test.hpp"
#include "dll/dbn_traits.hpp"

namespace dll {

/*!
 * \brief A generic trainer for Deep Belief Network
 *
 * This trainer use the specified trainer of the DBN to perform supervised
 * fine-tuning.
 */
template <typename DBN>
struct dbn_trainer {
    using dbn_t      = DBN;                    ///< The DBN type being trained
    using weight     = typename dbn_t::weight; ///< The data type for this layer
    using error_type = typename dbn_t::weight; ///< The error type

    /*!
     * \brief The trainer for the given RBM
     */
    template <typename R>
    using trainer_t = typename dbn_t::desc::template trainer_t<R>;

    /*!
     * \brief The watcher for the given RBM
     */
    template <typename R>
    using watcher_t = typename dbn_t::desc::template watcher_t<R>;

    //Initialize the watcher
    watcher_t<dbn_t> watcher; ///< The watcher for the DBN

    std::unique_ptr<trainer_t<dbn_t>> trainer; ///< The concrete trainer

    error_type error      = 0.0; ///< The current error

    template <typename Generator>
    error_type train(DBN& dbn, Generator& generator, size_t max_epochs) {
        return train_impl(dbn, generator, max_epochs);
    }

    /*!
     * \brief Initialize the training
     * \param dbn The network to train
     * \param ae Indicates if trained as auto-encoder or not
     * \param max_epochs How many epochs will be used
     */
    void start_training(dbn_t& dbn, size_t max_epochs){
        constexpr auto batch_size = std::decay_t<dbn_t>::batch_size;

        //Initialize the momentum
        dbn.momentum = dbn.initial_momentum;

        watcher.fine_tuning_begin(dbn, max_epochs);

        trainer = std::make_unique<trainer_t<dbn_t>>(dbn);

        //Initialize the trainer if necessary
        trainer->init_training(batch_size);

        // Set the initial error
        error = 0.0;
    }

    /*!
     * \brief Finalize the training
     * \param dbn The network that was trained
     */
    error_type stop_training(dbn_t& dbn){

        watcher.fine_tuning_end(dbn);

        return error;
    }

    /*!
     * \brief Start a new epoch
     * \param dbn The network that is trained
     * \param epoch The current epoch
     */
    void start_epoch(dbn_t& dbn, size_t epoch){
        watcher.ft_epoch_start(epoch, dbn);
    }

    /*!
     * \brief Indicates the end of an epoch
     * \param dbn The network that is trained
     * \param epoch The current epoch
     * \param new_error The new training error
     * \param loss The new training loss
     * \return true if the training is over
     */
    bool stop_epoch(dbn_t& dbn, size_t epoch, double new_error, double loss){
        auto last_error = new_error;

        error = new_error;

        //After some time increase the momentum
        if (dbn_traits<dbn_t>::has_momentum() && epoch == dbn.final_momentum_epoch) {
            dbn.momentum = dbn.final_momentum;
        }

        watcher.ft_epoch_end(epoch, new_error, loss, dbn);

        //Once the goal is reached, stop training
        if /*constexpr*/ (dbn_traits<dbn_t>::error_on_epoch()){
            if (new_error <= dbn.goal) {
                return true;
            }
        }

        if (dbn_traits<dbn_t>::lr_driver() == lr_driver_type::BOLD) {
            if (epoch) {
                if (new_error > last_error + 1e-8) {
                    //Error increased
                    dbn.learning_rate *= dbn.lr_bold_dec;
                    watcher.lr_adapt(dbn);
                    dbn.restore_weights();
                } else if (new_error < last_error - 1e-10) {
                    //Error decreased
                    dbn.learning_rate *= dbn.lr_bold_inc;
                    watcher.lr_adapt(dbn);
                    dbn.backup_weights();
                } else {
                    //Error didn't change enough
                    dbn.backup_weights();
                }
            } else {
                dbn.backup_weights();
            }
        }

        if (dbn_traits<dbn_t>::lr_driver() == lr_driver_type::STEP) {
            if (epoch && epoch % dbn.lr_step_size == 0) {
                dbn.learning_rate *= dbn.lr_step_gamma;
                watcher.lr_adapt(dbn);
            }
        }

        return false;
    }

    template<typename Generator>
    std::pair<double, double> train_epoch(dbn_t& dbn, Generator& generator, size_t epoch){
        // Set the generator in train mode
        generator.set_train();

        // Compute the number of batches
        const size_t batches = generator.batches();

        //Train one mini-batch at a time
        while(generator.has_next_batch()){
            dll::auto_timer timer("dbn::trainer::train_impl::epoch::batch");

            if /*constexpr*/ (dbn_traits<dbn_t>::is_verbose()){
                watcher.ft_batch_start(epoch, dbn);
            }

            double batch_error;
            double batch_loss;
            std::tie(batch_error, batch_loss) = trainer->train_batch(
                epoch,
                generator.data_batch(),
                generator.label_batch());

            if /*constexpr*/ (dbn_traits<dbn_t>::is_verbose()){
                watcher.ft_batch_end(epoch, generator.current_batch(), batches, batch_error, batch_loss, dbn);
            }

            generator.next_batch();
        }

        // Compute the error at this epoch
        double new_error;
        double new_loss;

        if /*constexpr*/ (dbn_traits<dbn_t>::error_on_epoch()){
            dll::auto_timer timer("dbn::trainer::train_impl::epoch::error");

            std::tie(new_error, new_loss) = dbn.evaluate_metrics(generator);
        } else {
            new_error = -1.0;
            new_loss  = -1.0;
        }

        return {new_loss, new_error};
    }

private:
    template <typename Generator>
    error_type train_impl(DBN& dbn, Generator& generator, size_t max_epochs) {
        dll::auto_timer timer("dbn::trainer::train_impl");

        // Initialization steps
        start_training(dbn, max_epochs);

        //Train the model for max_epochs epoch

        for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
            dll::auto_timer timer("dbn::trainer::train_impl::epoch");

            // Shuffle before the epoch if necessary
            if(dbn_traits<dbn_t>::shuffle()){
                generator.reset_shuffle();
            } else {
                generator.reset();
            }

            start_epoch(dbn, epoch);

            double new_error;
            double loss;
            std::tie(loss, new_error) = train_epoch(dbn, generator, epoch);

            if(stop_epoch(dbn, epoch, new_error, loss)){
                break;
            }
        }

        // Finalization

        return stop_training(dbn);
    }
};

} //end of dll namespace
