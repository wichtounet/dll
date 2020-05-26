//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <fstream>
#include <cstring>

#include <sys/stat.h>

#include "cpp_utils/stop_watch.hpp"

#include "trainer/rbm_training_context.hpp"
#include "layer_traits.hpp"
#include "dbn_traits.hpp"

namespace dll {

/*!
 * \brief The default watcher for RBM pretraining.
 * \tparam R The RBM type
 */
template <typename R>
struct default_rbm_watcher {
    cpp::stop_watch<std::chrono::seconds> full_timer; ///< Timer for the entire training
    dll::stop_timer epoch_timer;                      ///< Timer for an epoch

    /*!
     * \brief Indicates that the training of the given RBM started.
     * \param rbm The rbm that started training.
     */
    template <typename RBM = R>
    void training_begin(const RBM& rbm) {
        using rbm_t = std::decay_t<RBM>;

        std::cout << "Train RBM with \"" << RBM::desc::template trainer_t<RBM>::name() << "\"" << std::endl;

        rbm.display();

        std::cout << "With parameters:" << std::endl;

        if(std::is_same_v<typename rbm_t::weight, float>){
            std::cout << "   single-precision" << std::endl;
        } else if(std::is_same_v<typename rbm_t::weight, double>){
            std::cout << "   double-precision" << std::endl;
        } else {
            std::cout << "   unknown-precision (something is wrong...)" << std::endl;
        }

        std::cout << "   learning_rate=" << rbm.learning_rate << std::endl;
        std::cout << "   batch_size=" << RBM::batch_size << std::endl;

        if (rbm_layer_traits<RBM>::has_momentum()) {
            std::cout << "   momentum=" << rbm.momentum << std::endl;
        }

        if (rbm_layer_traits<RBM>::has_clip_gradients()) {
            std::cout << "   gradient clip=" << rbm.gradient_clip << std::endl;
        }

        if (w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1 || w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1L2) {
            std::cout << "   weight_cost(L1)=" << rbm.l1_weight_cost << std::endl;
        }

        if (w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L2 || w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1L2) {
            std::cout << "   weight_cost(L2)=" << rbm.l2_weight_cost << std::endl;
        }

        if (rbm_layer_traits<RBM>::sparsity_method() == sparsity_method::LEE) {
            std::cout << "   Sparsity (Lee): pbias=" << rbm.pbias << std::endl;
            std::cout << "   Sparsity (Lee): pbias_lambda=" << rbm.pbias_lambda << std::endl;
        } else if (rbm_layer_traits<RBM>::sparsity_method() == sparsity_method::GLOBAL_TARGET) {
            std::cout << "   sparsity_target(Global)=" << rbm.sparsity_target << std::endl;
        } else if (rbm_layer_traits<RBM>::sparsity_method() == sparsity_method::LOCAL_TARGET) {
            std::cout << "   sparsity_target(Local)=" << rbm.sparsity_target << std::endl;
        }

        std::cout << std::endl;
    }

    /*!
     * \brief Indicates the beginning of an epoch of pretraining.
     * \param epoch The epoch that just started training
     */
    void epoch_start([[maybe_unused]] size_t epoch) {
        epoch_timer.start();
    }

    /*!
     * \brief Indicates the end of an epoch of pretraining.
     * \param epoch The epoch that just finished training
     * \param context The RBM's training context
     * \param rbm The RBM being trained
     */
    template <typename RBM = R>
    void epoch_end(size_t epoch, const rbm_training_context& context, [[maybe_unused]] const RBM& rbm) {
        auto duration = epoch_timer.stop();

        char formatted[1024];
        if (rbm_layer_traits<RBM>::free_energy()) {
            snprintf(formatted, 1024, "epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f - Time: %ldms", epoch,
                     context.reconstruction_error, context.free_energy, context.sparsity, duration);
        } else {
            snprintf(formatted, 1024, "epoch %ld - Reconstruction error: %.5f - Sparsity: %.5f - Time: %ldms", epoch, context.reconstruction_error,
                     context.sparsity, duration);
        }

        std::cout << formatted << std::endl;
    }

    /*!
     * \brief Indicates the end of a batch of pretraining.
     * \param batch The batch that just finished training
     * \param batches The total number of batches
     * \param context The RBM's training context
     * \param rbm The RBM being trained
     */
    template <typename RBM = R>
    void batch_end([[maybe_unused]] const RBM& rbm, const rbm_training_context& context, size_t batch, size_t batches) {
        char formatted[1024];
        sprintf(formatted, "Batch %ld/%ld - Reconstruction error: %.5f - Sparsity: %.5f",
            batch, batches, context.batch_error, context.batch_sparsity);
        std::cout << formatted << std::endl;
    }

    /*!
     * \brief Indicates the end of pretraining.
     * \param rbm The RBM being trained
     */
    template <typename RBM = R>
    void training_end([[maybe_unused]] const RBM& rbm) {
        std::cout << "Training took " << full_timer.elapsed() << "s" << std::endl;
    }
};

/*!
 * \brief The default watcher for DBN training/pretraining
 */
template <typename DBN>
struct default_dbn_watcher {
    static constexpr bool ignore_sub  = false; ///< For pretraining of a DBN, indicates if the regular RBM watcher should be used (false) or ignored (true)
    static constexpr bool replace_sub = false; ///< For pretraining of a DBN, indicates if the DBN watcher should replace (true) the RBM watcher or not (false)

    size_t ft_max_epochs = 0;                         ///< The maximum number of epochs
    dll::stop_timer ft_epoch_timer;                   ///< Timer for an epoch
    dll::stop_timer ft_batch_timer;                   ///< Timer for a batch
    cpp::stop_watch<std::chrono::seconds> full_timer; ///< Timer for the entire training

    /*!
     * \brief Indicates that the pretraining has begun for the given
     * DBN
     * \param dbn The DBN being pretrained
     * \param max_epochs The maximum number of epochs
     */
    void pretraining_begin([[maybe_unused]] const DBN& dbn, size_t max_epochs) {
        std::cout << "DBN: Pretraining begin for " << max_epochs << " epochs" << std::endl;
    }

    /*!
     * \brief Indicates that the given layer is starting pretraining
     * \param dbn The DBN being trained
     * \param I The index of the layer being pretraining
     * \param rbm The RBM being trained
     * \param input_size the number of inputs
     */
    template <typename RBM>
    void pretrain_layer([[maybe_unused]] const DBN& dbn, size_t I, const RBM& rbm, size_t input_size) {
        if (input_size) {
            std::cout << "DBN: Pretrain layer " << I << " (" << rbm.to_full_string() << ") with " << input_size << " entries" << std::endl;
        } else {
            std::cout << "DBN: Pretrain layer " << I << " (" << rbm.to_full_string() << ")" << std::endl;
        }
    }

    /*!
     * \brief Indicates that the pretraining has ended for the given DBN
     * \param dbn The DBN being pretrained
     */
    void pretraining_end([[maybe_unused]] const DBN& dbn) {
        std::cout << "DBN: Pretraining finished after " << full_timer.elapsed() << "s" << std::endl;
    }

    /*!
     * \brief Pretraining ended for the given batch for the given DBN
     */
    void pretraining_batch([[maybe_unused]] const DBN& dbn, size_t batch) {
        std::cout << "DBN: Pretraining batch " << batch << std::endl;
    }

    /*!
     * \brief Fine-tuning of the given network just started
     * \param dbn The DBN that is being trained
     * \param max_epochs The maximum number of epochs to train the network
     */
    void fine_tuning_begin(const DBN& dbn, size_t max_epochs) {
        static constexpr auto UT = dbn_traits<DBN>::updater();

        std::cout << "\nTrain the network with \"" << DBN::desc::template trainer_t<DBN>::name() << "\"" << std::endl;
        std::cout << "    Updater: " << dll::to_string(UT) << std::endl;
        std::cout << "       Loss: " << dll::to_string(DBN::loss) << std::endl;
        std::cout << " Early Stop: " << dll::to_string(DBN::early) << std::endl << std::endl;

        std::cout << "With parameters:" << std::endl;
        std::cout << "          epochs=" << max_epochs << std::endl;
        std::cout << "      batch_size=" << DBN::batch_size << std::endl;

        // ADADELTA does not use the learning rate
        if (UT != updater_type::ADADELTA) {
            std::cout << "   learning_rate=" << dbn.learning_rate << std::endl;
        }

        if (UT == updater_type::MOMENTUM) {
            std::cout << "        momentum=" << dbn.momentum << std::endl;
        }

        if (UT == updater_type::NESTEROV) {
            std::cout << "        momentum=" << dbn.momentum << std::endl;
        }

        if (UT == updater_type::ADADELTA) {
            std::cout << "            beta=" << dbn.adadelta_beta << std::endl;
        }

        if (UT == updater_type::ADAM || UT == updater_type::ADAM_CORRECT || UT == updater_type::ADAMAX || UT == updater_type::NADAM) {
            std::cout << "           beta1=" << dbn.adam_beta1 << std::endl;
            std::cout << "           beta2=" << dbn.adam_beta2 << std::endl;
        }

        if (UT == updater_type::RMSPROP) {
            std::cout << "           decay=" << dbn.rmsprop_decay << std::endl;
        }

        if (w_decay(dbn_traits<DBN>::decay()) == decay_type::L1 || w_decay(dbn_traits<DBN>::decay()) == decay_type::L1L2) {
            std::cout << " weight_cost(L1)=" << dbn.l1_weight_cost << std::endl;
        }

        if (w_decay(dbn_traits<DBN>::decay()) == decay_type::L2 || w_decay(dbn_traits<DBN>::decay()) == decay_type::L1L2) {
            std::cout << " weight_cost(L2)=" << dbn.l2_weight_cost << std::endl;
        }

        std::cout << std::endl;

        ft_max_epochs = max_epochs;
    }

    size_t last_line_length = 0;
    size_t total_batch_duration = 0;
    size_t total_batches = 0;

    /*!
     * \brief One fine-tuning epoch is starting
     * \param epoch The current epoch
     * \param dbn The network being trained
     */
    void ft_epoch_start([[maybe_unused]] size_t epoch, [[maybe_unused]] const DBN& dbn) {
        ft_epoch_timer.start();

        last_line_length = 0;
    }

    /*!
     * \brief One fine-tuning epoch is over
     * \param epoch The current epoch
     * \param error The current error
     * \param loss The current loss
     * \param dbn The network being trained
     */
    void ft_epoch_end(size_t epoch, double error, double loss, [[maybe_unused]] const DBN& dbn) {
        auto duration = ft_epoch_timer.stop();

        char buffer[512];

        if (dbn_traits<DBN>::should_display_batch()) {
            if (dbn_traits<DBN>::error_on_epoch()){
                snprintf(buffer, 512, "epoch %3ld/%ld batch %4ld/%4ld - error: %.5f loss: %.5f time %ldms \n",
                    epoch, ft_max_epochs, max_batches, max_batches, error, loss, duration);
            } else {
                snprintf(buffer, 512, "epoch %3ld/%ld batch %4ld/%4ld - time %ldms \n",
                    epoch, ft_max_epochs, max_batches, max_batches, duration);
            }
        } else {
            if (dbn_traits<DBN>::error_on_epoch()){
                snprintf(buffer, 512, "epoch %3ld/%ld - error: %.5f loss: %.5f time %ldms \n",
                    epoch, ft_max_epochs, error, loss, duration);
            } else {
                snprintf(buffer, 512, "epoch %3ld/%ld - time %ldms \n",
                    epoch, ft_max_epochs, duration);
            }
        }

        if (dbn_traits<DBN>::is_verbose()){
            std::cout << buffer;
        } else {
            std::cout << "\r" << buffer;
        }

        std::cout.flush();
    }

    /*!
     * \brief One fine-tuning epoch is over
     * \param epoch The current epoch
     * \param train_error The current error
     * \param train_loss The current loss
     * \param dbn The network being trained
     */
    void ft_epoch_end(size_t epoch, double train_error, double train_loss, double val_error, double val_loss, [[maybe_unused]] const DBN& dbn) {
        auto duration = ft_epoch_timer.stop();

        char buffer[512];

        if constexpr (dbn_traits<DBN>::error_on_epoch()){
            snprintf(buffer, 512, "epoch %3ld/%ld - error: %.5f loss: %.5f val_error: %.5f val_loss: %.5f time %ldms \n",
                epoch, ft_max_epochs, train_error, train_loss, val_error, val_loss, duration);
        } else {
            snprintf(buffer, 512, "epoch %3ld/%ld - loss: %.5f val_loss: %.5f time %ldms \n",
                epoch, ft_max_epochs, train_loss, val_loss, duration);
        }

        if constexpr (dbn_traits<DBN>::is_verbose()){
            std::cout << buffer;
        } else {
            std::cout << "\r" << buffer;
        }

        std::cout.flush();
    }

    /*!
     * \brief Indicates the beginning of a fine-tuning batch
     * \param epoch The current epoch
     * \param dbn The DBN being trained
     */
    void ft_batch_start([[maybe_unused]] size_t epoch, [[maybe_unused]] const DBN& dbn) {
        ft_batch_timer.start();
    }

    size_t max_batches;

    /*!
     * \brief Indicates the end of a fine-tuning batch
     * \param epoch The current epoch
     * \param batch The current batch
     * \param batches THe total number of batches
     * \param batch_error The batch error
     * \param batch_loss The batch loss
     * \param dbn The DBN being trained
     */
    void ft_batch_end(size_t epoch, size_t batch, size_t batches, double batch_error, double batch_loss, [[maybe_unused]] const DBN& dbn) {
        auto duration = ft_batch_timer.stop();

        char buffer[512];

        if constexpr (dbn_traits<DBN>::is_verbose()){
            snprintf(buffer, 512, "epoch %3ld/%ld batch %4ld/%4ld- B. Error: %.5f B. Loss: %.5f Time %ldms",
                epoch, ft_max_epochs, batch + 1, batches, batch_error, batch_loss, duration);
            std::cout << buffer << std::endl;
        } else {
            total_batch_duration += duration;
            ++total_batches;

            auto estimated_duration = ((total_batch_duration / total_batches) * (batches - batch)) / 1000;
            snprintf(buffer, 512, "epoch %3ld/%ld batch %4ld/%4ld - error: %.5f loss: %.5f ETA %lds",
                epoch, ft_max_epochs, batch + 1, batches, batch_error, batch_loss, estimated_duration);

            if (batch == 0) {
                std::cout << buffer;
            } else {
                constexpr size_t frequency_ms = 100;

                if (total_batch_duration) {
                    const size_t frequency_batch = frequency_ms / (1 + (total_batch_duration / total_batches));

                    if (batch == batches - 1 || frequency_batch == 0 || batch % frequency_batch == 0) {
                        std::cout << "\r" << buffer;

                        if (strlen(buffer) < last_line_length) {
                            std::cout << std::string(last_line_length - strlen(buffer), ' ');
                        }

                        std::cout.flush();
                    }
                }
            }

            last_line_length = strlen(buffer);
        }

        max_batches = batches;
    }

    /*!
     * \brief Fine-tuning of the given network just finished
     * \param dbn The DBN that is being trained
     */
    void fine_tuning_end([[maybe_unused]] const DBN& dbn) {
        std::cout << "Training took " << full_timer.elapsed() << "s" << std::endl;
    }
};

template <typename DBN>
struct silent_dbn_watcher : default_dbn_watcher<DBN> {
    static constexpr bool ignore_sub  = true; ///< For pretraining of a DBN, indicates if the regular RBM watcher should be used (false) or ignored (true)
    static constexpr bool replace_sub = false; ///< For pretraining of a DBN, indicates if the DBN watcher should replace (true) the RBM watcher or not (false)
};

template <typename DBN>
struct mute_dbn_watcher {
    static constexpr bool ignore_sub  = true; ///< For pretraining of a DBN, indicates if the regular RBM watcher should be used (false) or ignored (true)
    static constexpr bool replace_sub = false; ///< For pretraining of a DBN, indicates if the DBN watcher should replace (true) the RBM watcher or not (false)

    /*!
     * \brief Indicates that the pretraining has begun for the given
     * DBN
     * \param dbn The DBN being pretrained
     * \param max_epochs The maximum number of epochs
     */
    void pretraining_begin([[maybe_unused]] const DBN& dbn, [[maybe_unused]] size_t max_epochs) {}

    /*!
     * \brief Indicates that the given layer is starting pretraining
     * \param dbn The DBN being trained
     * \param I The index of the layer being pretraining
     * \param rbm The RBM being trained
     * \param input_size the number of inputs
     */
    template <typename RBM>
    void pretrain_layer([[maybe_unused]] const DBN& dbn, [[maybe_unused]] size_t I, [[maybe_unused]] const RBM& rbm, [[maybe_unused]] size_t input_size) {}

    /*!
     * \brief Indicates that the pretraining has ended for the given DBN
     * \param dbn The DBN being pretrained
     */
    void pretraining_end([[maybe_unused]] const DBN& dbn) {}

    /*!
     * \brief Pretraining ended for the given batch for the given DBN
     */
    void pretraining_batch([[maybe_unused]] const DBN& dbn, [[maybe_unused]] size_t batch) {}

    /*!
     * \brief Fine-tuning of the given network just started
     * \param dbn The DBN that is being trained
     * \param max_epochs The maximum number of epochs to train the network
     */
    void fine_tuning_begin(const DBN& /*dbn*/, size_t /*max_epochs*/) {}

    /*!
     * \brief One fine-tuning epoch is starting
     * \param epoch The current epoch
     * \param dbn The network being trained
     */
    void ft_epoch_start([[maybe_unused]] size_t epoch, [[maybe_unused]] const DBN& dbn) {}

    /*!
     * \brief One fine-tuning epoch is ended
     * \param epoch The current epoch
     * \param error The current epoch error
     * \param dbn The network being trained
     */
    void ft_epoch_end([[maybe_unused]] size_t epoch, [[maybe_unused]] double error, [[maybe_unused]] double loss, [[maybe_unused]] const DBN& dbn) {}

    /*!
     * \brief One fine-tuning epoch is ended
     * \param epoch The current epoch
     * \param error The current epoch error
     * \param dbn The network being trained
     */
    void ft_epoch_end([[maybe_unused]] size_t epoch, [[maybe_unused]] double train_error, [[maybe_unused]] double train_loss, [[maybe_unused]] double val_error,
                      [[maybe_unused]] double val_loss, [[maybe_unused]] const DBN& dbn) {}

    /*!
     * \brief Indicates the beginning of a fine-tuning batch
     * \param epoch The current epoch
     * \param dbn The DBN being trained
     */
    void ft_batch_start(size_t /*epoch*/, const DBN&) {}

    /*!
     * \brief Indicates the end of a fine-tuning batch
     * \param epoch The current epoch
     * \param batch The current batch
     * \param batches THe total number of batches
     * \param batch_error The batch error
     * \param batch_loss The batch loss
     * \param dbn The DBN being trained
     */
    void ft_batch_end(size_t /*epoch*/, size_t /*batch*/, size_t /*batches*/, double /*batch_error*/, const DBN& /*dbn*/) {}

    void lr_adapt(const DBN& /*dbn*/) {}

    /*!
     * \brief Fine-tuning of the given network just finished
     * \param dbn The DBN that is being trained
     */
    void fine_tuning_end(const DBN& /*dbn*/) {}
};

} //end of dll namespace
