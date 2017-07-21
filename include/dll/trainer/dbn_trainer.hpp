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

    error_type current_error = 0.0; ///< The current training error
    error_type current_loss  = 0.0; ///< The current training loss

    error_type current_val_error = 0.0; ///< The current validation error
    error_type current_val_loss  = 0.0; ///< The current validation loss

    error_type best_error = 0.0; ///< The best error (training or validation depending on strategy)
    error_type best_loss  = 0.0; ///< The best loss (training or validation depending on strategy)
    size_t best_epoch     = 0;   ///< The best epoch

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

        // Set the initial error and loss
        current_error = 0.0;
        current_loss = 0.0;

        current_val_error = 0.0;
        current_val_loss = 0.0;
    }

    /*!
     * \brief Finalize the training
     * \param dbn The network that was trained
     */
    error_type stop_training(dbn_t& dbn){
        watcher.fine_tuning_end(dbn);

        return current_error;
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
     * \brief Decides to stop, or not, early the training.
     *
     * This function is also responsible to save the better weights according to
     * the early stopping strategy.
     *
     * \param dbn The network being trained
     * \param epoch The current epoch
     * \param error The current error
     * \param loss The current loss
     * \param prev_error The previous error
     * \param prev_loss The previous loss
     *
     * \return true if the training should be stopped, false otherwise
     */
    bool early_stop(dbn_t& dbn, size_t epoch, double error, double loss, double prev_error, double prev_loss){
        // The early stopping strategy
        static constexpr auto s = dbn_t::early;

        if /*constexpr*/ (s != strategy::NONE) {
            if /*constexpr*/ (is_error(s)) {
                if(!epoch || error < best_error){
                    best_error = error;
                    best_epoch = epoch;

                    dbn.backup_weights();
                }
            } else {
                if(!epoch || loss < best_loss){
                    best_loss = loss;
                    best_epoch = epoch;

                    dbn.backup_weights();
                }
            }
        }

        if /*constexpr*/ (dbn_traits<dbn_t>::error_on_epoch()) {
            // Stop according to goal on loss
            if /*constexpr*/ (s == strategy::LOSS_GOAL) {
                if (loss <= dbn.goal) {
                    std::cout << "Stopping: Loss below goal";

                    if(epoch != best_epoch){
                        dbn.restore_weights();

                        std::cout << ", restore weights from epoch " << best_epoch;
                    }

                    std::cout << std::endl;

                    return true;
                }
            }
            // Stop according to goal on error
            else if /*constexpr*/ (s == strategy::ERROR_GOAL) {
                if (error <= dbn.goal) {
                    std::cout << "Stopping: Error below goal";

                    if(epoch != best_epoch){
                        dbn.restore_weights();

                        std::cout << ", restore weights from epoch " << best_epoch;
                    }

                    std::cout << std::endl;

                    return true;
                }
            }
            // Stop if loss is increasing
            else if /*constexpr*/ (s == strategy::LOSS_DIRECT) {
                if (loss > prev_loss && epoch) {
                    std::cout << "Stopping: Loss is increasing";

                    if(epoch != best_epoch){
                        dbn.restore_weights();

                        std::cout << ", restore weights from epoch " << best_epoch;
                    }

                    std::cout << std::endl;

                    return true;
                }
            }
            // Stop if error is increasing
            else if /*constexpr*/ (s == strategy::ERROR_DIRECT) {
                if (error > prev_error && epoch) {
                    std::cout << "Stopping: Error is increasing";

                    if(epoch != best_epoch){
                        dbn.restore_weights();

                        std::cout << ", restore weights from epoch " << best_epoch;
                    }

                    std::cout << std::endl;

                    return true;
                }
            }
        }

        // Don't stop early
        return false;
    }

    /*!
     * \brief Indicates the end of an epoch
     * \param dbn The network that is trained
     * \param epoch The current epoch
     * \param error The new training error
     * \param loss The new training loss
     * \return true if the training is over
     */
    bool stop_epoch(dbn_t& dbn, size_t epoch, double error, double loss){
        //After some time increase the momentum
        if (dbn_traits<dbn_t>::updater() == updater_type::MOMENTUM && epoch == dbn.final_momentum_epoch) {
            dbn.momentum = dbn.final_momentum;
        }

        watcher.ft_epoch_end(epoch, error, loss, dbn);

        // Early stopping with training error/loss
        auto stop =  early_stop(dbn, epoch, error, loss, current_error, current_loss);

        // Save current error and loss
        current_error = error;
        current_loss  = loss;

        return stop;
    }

    /*!
     * \brief Indicates the end of an epoch
     * \param dbn The network that is trained
     * \param epoch The current epoch
     * \param new_error The new training error
     * \param loss The new training loss
     * \return true if the training is over
     */
    bool stop_epoch(dbn_t& dbn, size_t epoch, const std::pair<double, double>& train_stats, const std::pair<double, double>& val_stats){
        double error = train_stats.first;

        //After some time increase the momentum
        if (dbn_traits<dbn_t>::updater() == updater_type::MOMENTUM && epoch == dbn.final_momentum_epoch) {
            dbn.momentum = dbn.final_momentum;
        }

        watcher.ft_epoch_end(epoch, error, train_stats.second, val_stats.first, val_stats.second, dbn);

        // Early stopping with validation error/loss
        auto stop = early_stop(dbn, epoch, val_stats.first, val_stats.second, current_val_error, current_val_loss);

        // Save current error and loss for training and validation
        current_error = train_stats.first;
        current_loss  = train_stats.second;

        current_val_error = val_stats.first;
        current_val_loss  = val_stats.second;

        return stop;
    }

    template<typename Generator>
    std::pair<double, double> compute_error_loss(dbn_t& dbn, Generator& generator){
        // Compute the error and loss at this epoch
        double new_error;
        double new_loss;

        if /*constexpr*/ (dbn_traits<dbn_t>::error_on_epoch()){
            dll::auto_timer timer("dbn::trainer::train::epoch::error");

            std::tie(new_error, new_loss) = dbn.evaluate_metrics(generator);
        } else {
            new_error = -1.0;
            new_loss  = -1.0;
        }

        return std::make_pair(new_error, new_loss);
    }

    template<typename Generator>
    void train_epoch_only(dbn_t& dbn, Generator& generator, size_t epoch){
        // Set the generator in train mode
        generator.set_train();

        // Compute the number of batches
        const size_t batches = generator.batches();

        //Train one mini-batch at a time
        while(generator.has_next_batch()){
            dll::auto_timer timer("dbn::trainer::train::epoch::batch");

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
    }

    template<typename Generator>
    std::pair<double, double> train_epoch(dbn_t& dbn, Generator& generator, size_t epoch){
        // Train one epoch of training data
        train_epoch_only(dbn, generator, epoch);

        // Compute the error at this epoch
        return compute_error_loss(dbn, generator);
    }

    template<typename TrainGenerator, typename ValGenerator>
    std::pair<std::pair<double, double>, std::pair<double, double>> train_epoch(dbn_t& dbn, TrainGenerator& train_generator, ValGenerator& val_generator, size_t epoch){
        // Train one epoch of training data
        train_epoch_only(dbn, train_generator, epoch);

        // Compute the training error at this epoch
        auto train_stats = compute_error_loss(dbn, train_generator);

        // Compute the training error at this epoch
        auto val_stats = compute_error_loss(dbn, val_generator);

        // Return the stats
        return std::make_pair(train_stats, val_stats);
    }

    template <typename Generator>
    error_type train(DBN& dbn, Generator& generator, size_t max_epochs) {
        dll::auto_timer timer("dbn::trainer::train");

        // Initialization steps
        start_training(dbn, max_epochs);

        //Train the model for max_epochs epoch

        for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
            dll::auto_timer timer("dbn::trainer::train::epoch");

            // Shuffle before the epoch if necessary
            if(dbn_traits<dbn_t>::shuffle()){
                generator.reset_shuffle();
            } else {
                generator.reset();
            }

            start_epoch(dbn, epoch);

            double error;
            double loss;
            std::tie(error, loss) = train_epoch(dbn, generator, epoch);

            if(stop_epoch(dbn, epoch, error, loss)){
                break;
            }
        }

        // Finalization

        return stop_training(dbn);
    }

    template <typename TrainGenerator, typename ValGenerator>
    error_type train(DBN& dbn, TrainGenerator& train_generator, ValGenerator& val_generator, size_t max_epochs) {
        dll::auto_timer timer("dbn::trainer::train");

        // The validation generator is always in test mode
        val_generator.set_test();

        // Initialization steps
        start_training(dbn, max_epochs);

        //Train the model for max_epochs epoch

        for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
            dll::auto_timer timer("dbn::trainer::train::epoch");

            // Shuffle before the epoch if necessary
            if(dbn_traits<dbn_t>::shuffle()){
                train_generator.reset_shuffle();
            } else {
                train_generator.reset();
            }

            start_epoch(dbn, epoch);

            std::pair<double, double> train_stats;
            std::pair<double, double> val_stats;
            std::tie(train_stats, val_stats) = train_epoch(dbn, train_generator, val_generator, epoch);

            if (stop_epoch(dbn, epoch, train_stats, val_stats)) {
                break;
            }
        }

        // Finalization

        return stop_training(dbn);
    }
};

} //end of dll namespace
