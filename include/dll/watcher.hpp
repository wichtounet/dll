//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
    cpp::stop_watch<std::chrono::seconds> watch; ///< Timer for the entire training

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

        if(std::is_same<typename rbm_t::weight, float>::value){
            std::cout << "   single-precision" << std::endl;
        } else if(std::is_same<typename rbm_t::weight, double>::value){
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
     * \brief Indicates the end of an epoch of pretraining.
     * \param epoch The epoch that just finished training
     * \param context The RBM's training context
     * \param rbm The RBM being trained
     */
    template <typename RBM = R>
    void epoch_end(size_t epoch, const rbm_training_context& context, const RBM& rbm) {
        char formatted[1024];
        if (rbm_layer_traits<RBM>::free_energy()) {
            snprintf(formatted, 1024, "epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f", epoch,
                     context.reconstruction_error, context.free_energy, context.sparsity);
        } else {
            snprintf(formatted, 1024, "epoch %ld - Reconstruction error: %.5f - Sparsity: %.5f", epoch, context.reconstruction_error, context.sparsity);
        }

        std::cout << formatted << std::endl;

        cpp_unused(rbm);
    }

    /*!
     * \brief Indicates the end of a batch of pretraining.
     * \param batch The batch that just finished training
     * \param batches The total number of batches
     * \param context The RBM's training context
     * \param rbm The RBM being trained
     */
    template <typename RBM = R>
    void batch_end(const RBM& rbm, const rbm_training_context& context, size_t batch, size_t batches) {
        char formatted[1024];
        sprintf(formatted, "Batch %ld/%ld - Reconstruction error: %.5f - Sparsity: %.5f",
            batch, batches, context.batch_error, context.batch_sparsity);
        std::cout << formatted << std::endl;

        cpp_unused(rbm);
    }

    /*!
     * \brief Indicates the end of pretraining.
     * \param rbm The RBM being trained
     */
    template <typename RBM = R>
    void training_end(const RBM& rbm) {
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        cpp_unused(rbm);
    }
};

/*!
 * \brief The default watcher for DBN training/pretraining
 */
template <typename DBN>
struct default_dbn_watcher {
    static constexpr bool ignore_sub  = false; ///< For pretraining of a DBN, indicates if the regular RBM watcher should be used (false) or ignored (true)
    static constexpr bool replace_sub = false; ///< For pretraining of a DBN, indicates if the DBN watcher should replace (true) the RBM watcher or not (false)

    size_t ft_max_epochs = 0;                    ///< The maximum number of epochs
    dll::stop_timer ft_epoch_timer;              ///< Timer for an epoch
    dll::stop_timer ft_batch_timer;              ///< Timer for a batch
    cpp::stop_watch<std::chrono::seconds> watch; ///< Timer for the entire training

    /*!
     * \brief Indicates that the pretraining has begun for the given
     * DBN
     * \param dbn The DBN being pretrained
     * \param max_epochs The maximum number of epochs
     */
    void pretraining_begin(const DBN& dbn, size_t max_epochs) {
        *dbn.log << "DBN: Pretraining begin for " << max_epochs << " epochs" << std::endl;
        cpp_unused(dbn);
    }

    /*!
     * \brief Indicates that the given layer is starting pretraining
     * \param dbn The DBN being trained
     * \param I The index of the layer being pretraining
     * \param rbm The RBM being trained
     * \param input_size the number of inputs
     */
    template <typename RBM>
    void pretrain_layer(const DBN& dbn, size_t I, const RBM& rbm, size_t input_size) {
        if (input_size) {
            *dbn.log << "DBN: Pretrain layer " << I << " (" << rbm.to_full_string() << ") with " << input_size << " entries" << std::endl;
        } else {
            *dbn.log << "DBN: Pretrain layer " << I << " (" << rbm.to_full_string() << ")" << std::endl;
        }

        cpp_unused(dbn);
    }

    /*!
     * \brief Indicates that the pretraining has ended for the given DBN
     * \param dbn The DBN being pretrained
     */
    void pretraining_end(const DBN& dbn) {
        *dbn.log << "DBN: Pretraining finished after " << watch.elapsed() << "s" << std::endl;

        cpp_unused(dbn);
    }

    /*!
     * \brief Pretraining ended for the given batch for the given DBN
     */
    void pretraining_batch(const DBN& dbn, size_t batch) {
        *dbn.log << "DBN: Pretraining batch " << batch << std::endl;

        cpp_unused(dbn);
    }

    /*!
     * \brief Fine-tuning of the given network just started
     * \param dbn The DBN that is being trained
     * \param max_epochs The maximum number of epochs to train the network
     */
    void fine_tuning_begin(const DBN& dbn, size_t max_epochs) {
        static constexpr auto UT = dbn_traits<DBN>::updater();

        *dbn.log << "\nTrain the network with \"" << DBN::desc::template trainer_t<DBN>::name() << "\"" << std::endl;
        *dbn.log << "    Updater: " << dll::to_string(UT) << std::endl;
        *dbn.log << "       Loss: " << dll::to_string(DBN::loss) << std::endl;
        *dbn.log << " Early Stop: " << dll::to_string(DBN::early) << std::endl << std::endl;

        *dbn.log << "With parameters:" << std::endl;
        *dbn.log << "          epochs=" << max_epochs << std::endl;
        *dbn.log << "      batch_size=" << DBN::batch_size << std::endl;

        // ADADELTA does not use the learning rate
        if (UT != updater_type::ADADELTA) {
            *dbn.log << "   learning_rate=" << dbn.learning_rate << std::endl;
        }

        if (UT == updater_type::MOMENTUM) {
            *dbn.log << "        momentum=" << dbn.momentum << std::endl;
        }

        if (UT == updater_type::NESTEROV) {
            *dbn.log << "        momentum=" << dbn.momentum << std::endl;
        }

        if (UT == updater_type::ADADELTA) {
            *dbn.log << "            beta=" << dbn.adadelta_beta << std::endl;
        }

        if (UT == updater_type::ADAM || UT == updater_type::ADAM_CORRECT || UT == updater_type::ADAMAX || UT == updater_type::NADAM) {
            *dbn.log << "           beta1=" << dbn.adam_beta1 << std::endl;
            *dbn.log << "           beta2=" << dbn.adam_beta2 << std::endl;
        }

        if (UT == updater_type::RMSPROP) {
            *dbn.log << "           decay=" << dbn.rmsprop_decay << std::endl;
        }

        if (w_decay(dbn_traits<DBN>::decay()) == decay_type::L1 || w_decay(dbn_traits<DBN>::decay()) == decay_type::L1L2) {
            *dbn.log << " weight_cost(L1)=" << dbn.l1_weight_cost << std::endl;
        }

        if (w_decay(dbn_traits<DBN>::decay()) == decay_type::L2 || w_decay(dbn_traits<DBN>::decay()) == decay_type::L1L2) {
            *dbn.log << " weight_cost(L2)=" << dbn.l2_weight_cost << std::endl;
        }

        *dbn.log << std::endl;

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
    void ft_epoch_start(size_t epoch, const DBN& dbn) {
        cpp_unused(epoch);
        cpp_unused(dbn);
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
    void ft_epoch_end(size_t epoch, double error, double loss, const DBN& dbn) {
        cpp_unused(dbn);

        auto duration = ft_epoch_timer.stop();

        char buffer[512];

        if /*constexpr*/ (dbn_traits<DBN>::error_on_epoch()){
            snprintf(buffer, 512, "epoch %3ld/%ld batch %4ld/%4ld - error: %.5f loss: %.5f time %ldms \n",
                epoch, ft_max_epochs, max_batches, max_batches, error, loss, duration);
        } else {
            snprintf(buffer, 512, "epoch %3ld/%ld batch %4ld/%4ld - loss: %.5f time %ldms \n",
                epoch, ft_max_epochs, max_batches, max_batches, loss, duration);
        }

        if /*constexpr*/ (dbn_traits<DBN>::is_verbose()){
            *dbn.log << buffer;
        } else {
            *dbn.log << "\r" << buffer;
        }

        dbn.log->flush();
    }

    /*!
     * \brief One fine-tuning epoch is over
     * \param epoch The current epoch
     * \param train_error The current error
     * \param train_loss The current loss
     * \param dbn The network being trained
     */
    void ft_epoch_end(size_t epoch, double train_error, double train_loss, double val_error, double val_loss, const DBN& dbn) {
        cpp_unused(dbn);

        auto duration = ft_epoch_timer.stop();

        char buffer[512];

        if /*constexpr*/ (dbn_traits<DBN>::error_on_epoch()){
            snprintf(buffer, 512, "epoch %3ld/%ld - error: %.5f loss: %.5f val_error: %.5f val_loss: %.5f time %ldms \n",
                epoch, ft_max_epochs, train_error, train_loss, val_error, val_loss, duration);
        } else {
            snprintf(buffer, 512, "epoch %3ld/%ld - loss: %.5f val_loss: %.5f time %ldms \n",
                epoch, ft_max_epochs, train_loss, val_loss, duration);
        }

        if /*constexpr*/ (dbn_traits<DBN>::is_verbose()){
            *dbn.log << buffer;
        } else {
            *dbn.log << "\r" << buffer;
        }

        *dbn.log.flush();
    }

    /*!
     * \brief Indicates the beginning of a fine-tuning batch
     * \param epoch The current epoch
     * \param dbn The DBN being trained
     */
    void ft_batch_start(size_t epoch, const DBN& dbn) {
        cpp_unused(epoch);
        cpp_unused(dbn);
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
    void ft_batch_end(size_t epoch, size_t batch, size_t batches, double batch_error, double batch_loss, const DBN& dbn) {
        auto duration = ft_batch_timer.stop();

        char buffer[512];

        if /*constexpr*/ (dbn_traits<DBN>::is_verbose()){
            snprintf(buffer, 512, "epoch %3ld/%ld batch %4ld/%4ld- B. Error: %.5f B. Loss: %.5f Time %ldms",
                epoch, ft_max_epochs, batch + 1, batches, batch_error, batch_loss, duration);
            *dbn.log << buffer << std::endl;
        } else {
            total_batch_duration += duration;
            ++total_batches;

            auto estimated_duration = ((total_batch_duration / total_batches) * (batches - batch)) / 1000;
            snprintf(buffer, 512, "epoch %3ld/%ld batch %4ld/%4ld - error: %.5f loss: %.5f ETA %lds",
                epoch, ft_max_epochs, batch + 1, batches, batch_error, batch_loss, estimated_duration);

            if (batch == 0) {
                *dbn.log << buffer;
            } else {
                constexpr size_t frequency_ms = 100;

                if (total_batch_duration) {
                    const size_t frequency_batch = frequency_ms / (1 + (total_batch_duration / total_batches));

                    if (batch == batches - 1 || frequency_batch == 0 || batch % frequency_batch == 0) {
                        *dbn.log << "\r" << buffer;

                        if (strlen(buffer) < last_line_length) {
                            *dbn.log << std::string(last_line_length - strlen(buffer), ' ');
                        }

                        dbn.log->flush();
                    }
                }
            }

            last_line_length = strlen(buffer);
        }

        cpp_unused(dbn);

        max_batches = batches;
    }

    /*!
     * \brief Fine-tuning of the given network just finished
     * \param dbn The DBN that is being trained
     */
    void fine_tuning_end(const DBN& dbn) {
        *dbn.log << "Training took " << watch.elapsed() << "s" << std::endl;

        cpp_unused(dbn);
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
    void pretraining_begin(const DBN& dbn, size_t max_epochs) {
        cpp_unused(dbn);
        cpp_unused(max_epochs);
    }

    /*!
     * \brief Indicates that the given layer is starting pretraining
     * \param dbn The DBN being trained
     * \param I The index of the layer being pretraining
     * \param rbm The RBM being trained
     * \param input_size the number of inputs
     */
    template <typename RBM>
    void pretrain_layer(const DBN& dbn, size_t I, const RBM& rbm, size_t input_size) {
        cpp_unused(dbn);
        cpp_unused(I);
        cpp_unused(rbm);
        cpp_unused(input_size);
    }

    /*!
     * \brief Indicates that the pretraining has ended for the given DBN
     * \param dbn The DBN being pretrained
     */
    void pretraining_end(const DBN& dbn) {
        cpp_unused(dbn);
    }

    /*!
     * \brief Pretraining ended for the given batch for the given DBN
     */
    void pretraining_batch(const DBN& dbn, size_t batch) {
        cpp_unused(dbn);
        cpp_unused(batch);
    }

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
    void ft_epoch_start(size_t epoch, const DBN& dbn) {
        cpp_unused(epoch);
        cpp_unused(dbn);
    }

    /*!
     * \brief One fine-tuning epoch is ended
     * \param epoch The current epoch
     * \param error The current epoch error
     * \param dbn The network being trained
     */
    void ft_epoch_end(size_t epoch, double error, double loss, const DBN& dbn) {
        cpp_unused(epoch);
        cpp_unused(error);
        cpp_unused(loss);
        cpp_unused(dbn);
    }

    /*!
     * \brief One fine-tuning epoch is ended
     * \param epoch The current epoch
     * \param error The current epoch error
     * \param dbn The network being trained
     */
    void ft_epoch_end(size_t epoch, double train_error, double train_loss, double val_error, double val_loss, const DBN& dbn) {
        cpp_unused(epoch);
        cpp_unused(train_error);
        cpp_unused(train_loss);
        cpp_unused(val_error);
        cpp_unused(val_loss);
        cpp_unused(dbn);
    }

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
