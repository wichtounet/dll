//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <fstream>

#include <sys/stat.h>

#include "cpp_utils/stop_watch.hpp"

#include "trainer/rbm_training_context.hpp"
#include "layer_traits.hpp"
#include "dbn_traits.hpp"

namespace dll {

template <typename R>
struct default_rbm_watcher {
    cpp::stop_watch<std::chrono::seconds> watch;

    template <typename RBM = R>
    void training_begin(const RBM& rbm) {
        using rbm_t = std::decay_t<RBM>;

        std::cout << "Train RBM with \"" << RBM::desc::template trainer_t<RBM, false>::name() << "\"" << std::endl;

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
        std::cout << "   batch_size=" << get_batch_size(rbm) << std::endl;

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
    }

    template <typename RBM = R>
    void epoch_end(std::size_t epoch, const rbm_training_context& context, const RBM& /*rbm*/) {
        char formatted[1024];
        if (rbm_layer_traits<RBM>::free_energy()) {
            snprintf(formatted, 1024, "epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f", epoch,
                     context.reconstruction_error, context.free_energy, context.sparsity);
        } else {
            snprintf(formatted, 1024, "epoch %ld - Reconstruction error: %.5f - Sparsity: %.5f", epoch, context.reconstruction_error, context.sparsity);
        }
        std::cout << formatted << std::endl;
    }

    template <typename RBM = R>
    void batch_end(const RBM& /* rbm */, const rbm_training_context& context, std::size_t batch, std::size_t batches) {
        char formatted[1024];
        sprintf(formatted, "Batch %ld/%ld - Reconstruction error: %.5f - Sparsity: %.5f", batch, batches,
                context.batch_error, context.batch_sparsity);
        std::cout << formatted << std::endl;
    }

    template <typename RBM = R>
    void training_end(const RBM&) {
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;
    }
};

template <typename DBN>
struct default_dbn_watcher {
    static constexpr bool ignore_sub  = false;
    static constexpr bool replace_sub = false;

    size_t ft_max_epochs = 0;
    dll::stop_timer ft_epoch_timer;
    dll::stop_timer ft_batch_timer;

    cpp::stop_watch<std::chrono::seconds> watch;

    void pretraining_begin(const DBN& /*dbn*/, std::size_t max_epochs) {
        std::cout << "DBN: Pretraining begin for " << max_epochs << " epochs" << std::endl;
    }

    template <typename RBM>
    void pretrain_layer(const DBN& /*dbn*/, std::size_t I, const RBM& rbm, std::size_t input_size) {
        if (input_size) {
            std::cout << "DBN: Pretrain layer " << I << " (" << rbm.to_short_string() << ") with " << input_size << " entries" << std::endl;
        } else {
            std::cout << "DBN: Pretrain layer " << I << " (" << rbm.to_short_string() << ")" << std::endl;
        }
    }

    void pretraining_end(const DBN& /*dbn*/) {
        std::cout << "DBN: Pretraining finished after " << watch.elapsed() << "s" << std::endl;
    }

    void pretraining_batch(const DBN& /*dbn*/, std::size_t batch) {
        std::cout << "DBN: Pretraining batch " << batch << std::endl;
    }

    /*!
     * \brief Fine-tuning of the given network just started
     * \param dbn The DBN that is being trained
     * \param max_epochs The maximum number of epochs to train the network
     */
    void fine_tuning_begin(const DBN& dbn, size_t max_epochs) {
        std::cout << "Train the network with \"" << DBN::desc::template trainer_t<DBN>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "          epochs=" << max_epochs << std::endl;
        std::cout << "      batch_size=" << DBN::batch_size << std::endl;
        std::cout << "   learning_rate=" << dbn.learning_rate << std::endl;

        if (dbn_traits<DBN>::has_momentum()) {
            std::cout << "   momentum=" << dbn.momentum << std::endl;
        }

        if (w_decay(dbn_traits<DBN>::decay()) == decay_type::L1 || w_decay(dbn_traits<DBN>::decay()) == decay_type::L1L2) {
            std::cout << "   weight_cost(L1)=" << dbn.l1_weight_cost << std::endl;
        }

        if (w_decay(dbn_traits<DBN>::decay()) == decay_type::L2 || w_decay(dbn_traits<DBN>::decay()) == decay_type::L1L2) {
            std::cout << "   weight_cost(L2)=" << dbn.l2_weight_cost << std::endl;
        }

        if (dbn_traits<DBN>::lr_driver() == lr_driver_type::BOLD) {
            std::cout << "   lr_driver(BOLD)=" << dbn.lr_bold_inc << ":" << dbn.lr_bold_dec << std::endl;
        }

        if (dbn_traits<DBN>::lr_driver() == lr_driver_type::STEP) {
            std::cout << "   lr_driver(STEP)=" << dbn.lr_step_size << ":" << dbn.lr_step_gamma << std::endl;
        }

        ft_max_epochs = max_epochs;
    }

    /*!
     * \brief One fine-tuning epoch is starting
     * \param epoch The current epoch
     * \param dbn The network being trained
     */
    void ft_epoch_start(std::size_t epoch, const DBN& dbn) {
        cpp_unused(epoch);
        cpp_unused(dbn);
        ft_epoch_timer.start();
    }

    /*!
     * \brief One fine-tuning epoch is over
     * \param epoch The current epoch
     * \param error The current error
     * \param loss The current loss
     * \param dbn The network being trained
     */
    void ft_epoch_end(std::size_t epoch, double error, double loss, const DBN& dbn) {
        cpp_unused(dbn);
        auto duration = ft_epoch_timer.stop();

        if /*constexpr*/ (dbn_traits<DBN>::error_on_epoch()){
            printf("Epoch %3ld/%ld - Classification error: %.5f Loss: %.5f Time %ldms \n", epoch, ft_max_epochs, error, loss, duration);
        } else {
            printf("Epoch %3ld/%ld - Loss: %.5f Time %ldms \n", epoch, ft_max_epochs, loss, duration);
        }

        std::cout.flush();
    }

    void ft_batch_start(size_t epoch, const DBN& dbn) {
        cpp_unused(epoch);
        cpp_unused(dbn);
        ft_batch_timer.start();
    }

    void ft_batch_end(size_t epoch, size_t batch, size_t batches, double batch_error, double batch_loss, const DBN&) {
        auto duration = ft_batch_timer.stop();
        printf("Epoch %3ld:%ld/%ld- B. Error: %.5f B. Loss: %.5f Time %ldms\n", epoch, batch, batches, batch_error, batch_loss, duration);
        std::cout.flush();
    }

    void ft_batch_end(size_t epoch, double batch_error, double batch_loss, const DBN&) {
        auto duration = ft_batch_timer.stop();
        printf("Epoch %3ld - B.Error: %.5f B.Loss: %.5f Time %ldms\n", epoch, batch_error, batch_loss, duration);
        std::cout.flush();
    }

    void lr_adapt(const DBN& dbn) {
        printf("driver: learning rate adapted to %.5f \n", dbn.learning_rate);
        std::cout.flush();
    }

    void fine_tuning_end(const DBN&) {
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;
    }
};

template <typename DBN>
struct silent_dbn_watcher : default_dbn_watcher<DBN> {
    static constexpr bool ignore_sub  = true;
    static constexpr bool replace_sub = false;
};

template <typename DBN>
struct mute_dbn_watcher {
    static constexpr bool ignore_sub  = true;
    static constexpr bool replace_sub = false;

    void pretraining_begin(const DBN& /*dbn*/, std::size_t /*max_epochs*/) {}

    template <typename RBM>
    void pretrain_layer(const DBN& /*dbn*/, std::size_t /*I*/, const RBM& /*rbm*/, std::size_t /*input_size*/) {}

    void pretraining_end(const DBN& /*dbn*/) {}

    void pretraining_batch(const DBN& /*dbn*/, std::size_t /*batch*/) {}

    void fine_tuning_begin(const DBN& /*dbn*/, size_t /*max_epochs*/) {}

    void ft_epoch_start(std::size_t /*epoch*/, const DBN& /*dbn*/) {}

    void ft_epoch_end(std::size_t /*epoch*/, double /*error*/, const DBN& /*dbn*/) {}

    void ft_batch_start(size_t /*epoch*/, const DBN&) {}
    void ft_batch_end(size_t /*epoch*/, size_t /*batch*/, size_t /*batches*/, double /*batch_error*/, const DBN& /*dbn*/) {}
    void ft_batch_end(size_t /*epoch*/, double /*batch_error*/, const DBN& /*dbn*/) {}

    void lr_adapt(const DBN& /*dbn*/) {}

    void fine_tuning_end(const DBN& /*dbn*/) {}
};

//TODO This is currently useless

template <typename R>
struct histogram_watcher {
    default_rbm_watcher<R> parent;

    template <typename RBM = R>
    void training_begin(const RBM& rbm) {
        parent.training_begin(rbm);
    }

    template <typename RBM = R>
    void epoch_end(std::size_t epoch, double error, double /*free_energy*/, const RBM& rbm) {
        parent.epoch_end(epoch, error, rbm);
    }

    template <typename RBM = R>
    void batch_end(const RBM& rbm, const rbm_training_context& context, std::size_t batch, std::size_t batches) {
        parent.batch_end(rbm, context, batch, batches);
    }

    template <typename RBM = R>
    void training_end(const RBM& rbm) {
        parent.training_end(rbm);
    }

    void generate_hidden_images(std::size_t epoch, const R& rbm) {
        mkdir("reports", 0777);

        auto folder = "reports/epoch_" + std::to_string(epoch);
        mkdir(folder.c_str(), 0777);

        for (std::size_t j = 0; j < R::num_hidden; ++j) {
            auto path = folder + "/h_" + std::to_string(j) + ".dat";
            std::ofstream file(path, std::ios::out);

            if (!file) {
                std::cout << "Could not open file " << path << std::endl;
            } else {
                std::size_t i = R::num_visible;
                while (i > 0) {
                    --i;

                    auto value = rbm.w(i, j);
                    file << static_cast<std::size_t>(value > 0 ? static_cast<std::size_t>(value * 255.0) << 8 : static_cast<std::size_t>(-value * 255.0) << 16) << " ";
                }

                file << std::endl;
                file.close();
            }
        }
    }

    void generate_histograms(std::size_t epoch, const R& rbm) {
        mkdir("reports", 0777);

        auto folder = "reports/epoch_" + std::to_string(epoch);
        mkdir(folder.c_str(), 0777);

        generate_histogram(folder + "/weights.dat", rbm.w);
        generate_histogram(folder + "/visibles.dat", rbm.a);
        generate_histogram(folder + "/hiddens.dat", rbm.b);
    }

    template <typename Container>
    void generate_histogram(const std::string& path, const Container& weights) {
        std::ofstream file(path, std::ios::out);

        if (!file) {
            std::cout << "Could not open file " << path << std::endl;
        } else {
            for (auto& weight : weights) {
                file << weight << std::endl;
            }

            file << std::endl;
            file.close();
        }
    }
};

} //end of dll namespace
