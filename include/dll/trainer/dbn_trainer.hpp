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
    size_t patience       = 0;   ///< The current patience

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
     *
     * \param dbn The network that was trained
     * \param epoch The current epoch
     * \param max_epochs
     *
     * \return the final error
     */
    error_type stop_training(dbn_t& dbn, size_t epoch, size_t max_epochs){
        // Depending on the strategy, try to restore the best weights

        if(epoch == max_epochs){
            // The early stopping strategy
            static constexpr auto s = dbn_t::early;

            if /*constexpr*/ (s != strategy::NONE) {
                if(best_epoch < max_epochs - 1){
                    dbn.restore_weights();

                    if (is_error(s)) {
                        *dbn.log << "Restore the best (error) weights from epoch " << best_epoch << std::endl;
                    } else {
                        *dbn.log << "Restore the best (loss) weights from epoch " << best_epoch << std::endl;
                    }
                }
            }
        }

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

        // Depending on the strategy, try to save the best weights

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

        // Depending on the strategy, decide to stop training

        if /*constexpr*/ (dbn_traits<dbn_t>::error_on_epoch()) {
            // Stop according to goal on loss
            if /*constexpr*/ (s == strategy::LOSS_GOAL) {
                if (loss <= dbn.goal) {
                    *dbn.log << "Stopping: Loss below goal";

                    if(epoch != best_epoch){
                        dbn.restore_weights();

                        *dbn.log << ", restore weights from epoch " << best_epoch;
                    }

                    *dbn.log << std::endl;

                    return true;
                }
            }
            // Stop according to goal on error
            else if /*constexpr*/ (s == strategy::ERROR_GOAL) {
                if (error <= dbn.goal) {
                    *dbn.log << "Stopping: Error below goal";

                    if(epoch != best_epoch){
                        dbn.restore_weights();

                        *dbn.log << ", restore weights from epoch " << best_epoch;
                    }

                    *dbn.log << std::endl;

                    return true;
                }
            }
            // Stop if loss is increasing
            else if /*constexpr*/ (s == strategy::LOSS_DIRECT) {
                if (loss > prev_loss && epoch) {
                    --patience;

                    if (!patience) {
                        *dbn.log << "Stopping: Loss has been increasing for " << dbn.patience << " epochs";

                        if (epoch != best_epoch) {
                            dbn.restore_weights();

                            *dbn.log << ", restore weights from epoch " << best_epoch;
                        }

                        *dbn.log << std::endl;

                        return true;
                    }
                } else {
                    patience = dbn.patience;
                }
            }
            // Stop if error is increasing
            else if /*constexpr*/ (s == strategy::ERROR_DIRECT) {
                if (error > prev_error && epoch) {
                    --patience;

                    if (!patience) {
                        *dbn.log << "Stopping: Error has been increasing for " << dbn.patience << " epochs";

                        if (epoch != best_epoch) {
                            dbn.restore_weights();

                            *dbn.log << ", restore weights from epoch " << best_epoch;
                        }

                        *dbn.log << std::endl;

                        return true;
                    }
                } else {
                    patience = dbn.patience;
                }
            }
            // Stop if loss is increasing (relative to best)
            else if /*constexpr*/ (s == strategy::LOSS_BEST) {
                if (loss > best_loss && epoch) {
                    --patience;

                    if (!patience) {
                        *dbn.log << "Stopping: Loss has been increasing (from best) for " << dbn.patience << " epochs";

                        if (epoch != best_epoch) {
                            dbn.restore_weights();

                            *dbn.log << ", restore weights from epoch " << best_epoch;
                        }

                        *dbn.log << std::endl;

                        return true;
                    }
                } else {
                    patience = dbn.patience;
                }
            }
            // Stop if error is increasing (relative to best)
            else if /*constexpr*/ (s == strategy::ERROR_BEST) {
                if (error > best_error && epoch) {
                    --patience;

                    if (!patience) {
                        *dbn.log << "Stopping: Error has been increasing (from best) for " << dbn.patience << " epochs";

                        if (epoch != best_epoch) {
                            dbn.restore_weights();

                            *dbn.log << ", restore weights from epoch " << best_epoch;
                        }

                        *dbn.log << std::endl;

                        return true;
                    }
                } else {
                    patience = dbn.patience;
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

        // Early stopping with validation (or training) error/loss

        bool stop;
        if (dbn_traits<dbn_t>::early_uses_training()) {
            stop = early_stop(dbn, epoch, train_stats.first, train_stats.second, current_error, current_loss);
        } else {
            stop = early_stop(dbn, epoch, val_stats.first, val_stats.second, current_val_error, current_val_loss);
        }

        // Save current error and loss for training and validation
        current_error = train_stats.first;
        current_loss  = train_stats.second;

        current_val_error = val_stats.first;
        current_val_loss  = val_stats.second;

        return stop;
    }

    /*!
     * \brief Compute error and loss on the given generator with the given network.
     * \param dbn The network to be used
     * \param generator The generator to get data from
     * \return a pair containing (error, loss)
     */
    template<typename Generator>
    std::pair<double, double> compute_error_loss(dbn_t& dbn, Generator& generator){
        // Compute the error and loss at this epoch
        double new_error =  1.0;
        double new_loss  = -1.0;

        if /*constexpr*/ (dbn_traits<dbn_t>::error_on_epoch()){
            dll::auto_timer timer("net:trainer:train:epoch:error");

            auto forward_helper = [this, &dbn](auto&& input_batch) -> decltype(auto) {
                return this->trainer->template forward_batch_helper<false>(dbn, input_batch);
            };

            std::tie(new_error, new_loss) = dbn.evaluate_metrics(generator, forward_helper);
        }

        return std::make_pair(new_error, new_loss);
    }

    /*!
     * \brief Train the network for one epoch
     * \param generator The generator for training data
     * \param epoch The current epoch
     */
    template<typename Generator>
    void train_epoch_only(dbn_t& dbn, Generator& generator, size_t epoch){
        // Set the generator in train mode
        generator.set_train();

        //Train one mini-batch at a time
        while(generator.has_next_batch()){
            dll::auto_timer timer("net:trainer:train:epoch:batch");

            watcher.ft_batch_start(epoch, dbn);

            double batch_error;
            double batch_loss;
            std::tie(batch_error, batch_loss) = trainer->train_batch(
                epoch,
                generator.data_batch(),
                generator.label_batch());

            watcher.ft_batch_end(epoch, generator.current_batch(), generator.batches(), batch_error, batch_loss, dbn);

            generator.next_batch();
        }
    }

    /*!
     * \brief Train the network for one epoch and compute the loss and error on the training set
     *
     * \param dbn The network to train
     * \param generator The generator to use for training data
     * \param epoch The current epoch
     *
     * \return a pair containing (error, loss)
     */
    template<typename Generator>
    std::pair<double, double> train_epoch(dbn_t& dbn, Generator& generator, size_t epoch){
        // Train one epoch of training data
        train_epoch_only(dbn, generator, epoch);

        // Compute the error at this epoch
        return compute_error_loss(dbn, generator);
    }

    /*!
     * \brief Train the network for one epoch and compute the loss and error
     * on the training and validation sets
     *
     * \param dbn The network to train
     * \param train_generator The generator to use for training data
     * \param val_generator The generator to use for validation data
     * \param epoch The current epoch
     *
     * \return a pair of pair containing ((train_error, train_loss), (val_error, val_loss))
     */
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

    /*!
     * \brief Train the network for max_epochs
     *
     * \param dbn The network to be trained
     * \param generator The generator for the training data
     * \param max_epochs The maximum number of epochs
     *
     * \return The final error
     */
    template <typename Generator>
    error_type train(DBN& dbn, Generator& generator, size_t max_epochs) {
        dll::auto_timer timer("net:trainer:train");

        // Initialization steps
        start_training(dbn, max_epochs);

        //Train the model for max_epochs epoch

        size_t epoch = 0;
        for (; epoch < max_epochs; ++epoch) {
            dll::auto_timer timer("net:trainer:train:epoch");

            {
                dll::auto_timer timer("net:trainer:train:epoch:prepare");

                // Shuffle before the epoch if necessary
                if(dbn_traits<dbn_t>::shuffle()){
                    generator.reset_shuffle();
                } else {
                    generator.reset();
                }

                // This will ensure maximum performance for the training
                generator.prepare_epoch();
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

        return stop_training(dbn, epoch, max_epochs);
    }

    /*!
     * \brief Train the network for max_epochs
     *
     * \param dbn The network to be trained
     * \param train_generator The generator for the training data
     * \param val_generator The generator for the validation data
     * \param max_epochs The maximum number of epochs
     *
     * \return The final error
     */
    template <typename TrainGenerator, typename ValGenerator>
    error_type train(DBN& dbn, TrainGenerator& train_generator, ValGenerator& val_generator, size_t max_epochs) {
        dll::auto_timer timer("net:trainer:train");

        // The validation generator is always in test mode
        val_generator.set_test();

        // Initialization steps
        start_training(dbn, max_epochs);

        //Train the model for max_epochs epoch

        size_t epoch = 0;
        for (; epoch < max_epochs; ++epoch) {
            dll::auto_timer timer("net:trainer:train:epoch");

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

        return stop_training(dbn, epoch, max_epochs);
    }
};

} //end of dll namespace
