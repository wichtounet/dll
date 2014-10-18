//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_WATCHER_HPP
#define DLL_WATCHER_HPP

#include <fstream>

#include <sys/stat.h>

#include "cpp_utils/stop_watch.hpp"

#include "rbm_traits.hpp"
#include "dbn_traits.hpp"

namespace dll {

template<typename R>
struct default_rbm_watcher {
    cpp::stop_watch<std::chrono::seconds> watch;

    template<typename RBM = R>
    void training_begin(const RBM& rbm){
        std::cout << "Train RBM with \"" << RBM::desc::template trainer_t<RBM>::name() << "\"" << std::endl;

        rbm.display();

        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << rbm.learning_rate << std::endl;

        if(rbm_traits<RBM>::has_momentum()){
            std::cout << "   momentum=" << rbm.momentum << std::endl;
        }

        if(w_decay(rbm_traits<RBM>::decay()) == decay_type::L1 || w_decay(rbm_traits<RBM>::decay()) == decay_type::L1L2){
            std::cout << "   weight_cost(L1)=" << rbm.l1_weight_cost << std::endl;
        }

        if(w_decay(rbm_traits<RBM>::decay()) == decay_type::L2 || w_decay(rbm_traits<RBM>::decay()) == decay_type::L1L2){
            std::cout << "   weight_cost(L2)=" << rbm.l2_weight_cost << std::endl;
        }

        if(rbm_traits<RBM>::sparsity_method() == sparsity_method::LEE){
            std::cout << "   Sparsity (Lee): pbias=" << rbm.pbias << std::endl;
            std::cout << "   Sparsity (Lee): pbias_lambda=" << rbm.pbias_lambda << std::endl;
        } else if(rbm_traits<RBM>::sparsity_method() == sparsity_method::GLOBAL_TARGET){
            std::cout << "   sparsity_target(Global)=" << rbm.sparsity_target << std::endl;
        } else if(rbm_traits<RBM>::sparsity_method() == sparsity_method::LOCAL_TARGET){
            std::cout << "   sparsity_target(Local)=" << rbm.sparsity_target << std::endl;
        }
    }

    template<typename RBM = R>
    void epoch_end(std::size_t epoch, const rbm_training_context& context, const RBM& /*rbm*/){
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
            context.reconstruction_error, context.free_energy, context.sparsity);
    }

    template<typename RBM = R>
    void training_end(const RBM&){
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;
    }
};

template<typename DBN, typename Enable = void>
struct default_dbn_watcher {
    static constexpr const bool ignore_sub = false;
    static constexpr const bool replace_sub = false;

    cpp::stop_watch<std::chrono::seconds> watch;

    void pretraining_begin(const DBN& /*dbn*/){
        std::cout << "DBN: Pretraining begin" << std::endl;
    }

    template<typename RBM>
    void pretrain_layer(const DBN& /*dbn*/, std::size_t I, std::size_t input_size){
        using rbm_t = RBM;
        static constexpr const auto num_visible = rbm_t::num_visible;
        static constexpr const auto num_hidden = rbm_t::num_hidden;

        std::cout << "DBN: Train layer " << I << " (" << num_visible << "->" << num_hidden << ") with " << input_size << " entries" << std::endl;
    }

    void pretraining_end(const DBN& /*dbn*/){
        std::cout << "DBN: Pretraining end" << std::endl;
    }

    void fine_tuning_begin(const DBN& dbn){
        std::cout << "Train DBN with \"" << DBN::desc::template trainer_t<DBN>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << dbn.learning_rate << std::endl;

        if(dbn_traits<DBN>::has_momentum()){
            std::cout << "   momentum=" << dbn.momentum << std::endl;
        }
    }

    void ft_epoch_end(std::size_t epoch, double error, const DBN&){
        printf("epoch %ld - Classification error: %.5f \n", epoch, error);
    }

    void fine_tuning_end(const DBN&){
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;
    }
};

template<typename DBN>
struct default_dbn_watcher<DBN, std::enable_if_t<dbn_traits<DBN>::is_convolutional()>> {
    static constexpr const bool ignore_sub = false;
    static constexpr const bool replace_sub = false;

    cpp::stop_watch<std::chrono::seconds> watch;

    void pretraining_begin(const DBN& /*dbn*/){
        std::cout << "CDBN: Pretraining begin" << std::endl;
    }

    template<typename RBM>
    void pretrain_layer(const DBN& /*dbn*/, std::size_t I, std::size_t input_size){
        using rbm_t = RBM;

        static constexpr const auto NV = rbm_t::NV;
        static constexpr const auto NH = rbm_t::NH;
        static constexpr const auto K = rbm_t::K;

        printf("CDBN: Train layer %lu (%lux%lu -> %lux%lu (%lu)) with %lu entries \n", I, NV, NV, NH, NH, K, input_size);
    }

    void pretraining_end(const DBN& /*dbn*/){
        std::cout << "CDBN: Pretraining end" << std::endl;
    }
};

template<typename DBN>
struct silent_dbn_watcher : default_dbn_watcher<DBN> {
    static constexpr const bool ignore_sub = true;
    static constexpr const bool replace_sub = false;
};

//TODO This is currently useless

template<typename R>
struct histogram_watcher {
    default_rbm_watcher<R> parent;

    template<typename RBM = R>
    void training_begin(const RBM& rbm){
        parent.training_begin(rbm);
    }

    template<typename RBM = R>
    void epoch_end(std::size_t epoch, double error, double /*free_energy*/, const RBM& rbm){
        parent.epoch_end(epoch, error, rbm);
    }

    template<typename RBM = R>
    void training_end(const RBM& rbm){
        parent.training_end(rbm);
    }

    void generate_hidden_images(std::size_t epoch, const R& rbm){
        mkdir("reports", 0777);

        auto folder = "reports/epoch_" + std::to_string(epoch);
        mkdir(folder.c_str(), 0777);

        for(std::size_t j = 0; j < R::num_hidden; ++j){
            auto path = folder + "/h_" + std::to_string(j) + ".dat";
            std::ofstream file(path, std::ios::out);

            if(!file){
                std::cout << "Could not open file " << path << std::endl;
            } else {
                std::size_t i = R::num_visible;
                while(i > 0){
                    --i;

                    auto value = rbm.w(i,j);
                    file << static_cast<std::size_t>(value > 0 ? static_cast<std::size_t>(value * 255.0) << 8 : static_cast<std::size_t>(-value * 255.0) << 16) << " ";
                }

                file << std::endl;
                file.close();
            }
        }
    }

    void generate_histograms(std::size_t epoch, const R& rbm){
        mkdir("reports", 0777);

        auto folder = "reports/epoch_" + std::to_string(epoch);
        mkdir(folder.c_str(), 0777);

        generate_histogram(folder + "/weights.dat", rbm.w);
        generate_histogram(folder + "/visibles.dat", rbm.a);
        generate_histogram(folder + "/hiddens.dat", rbm.b);
    }

    template<typename Container>
    void generate_histogram(const std::string& path, const Container& weights){
        std::ofstream file(path, std::ios::out);

        if(!file){
            std::cout << "Could not open file " << path << std::endl;
        } else {
            for(auto& weight : weights){
                file << weight << std::endl;
            }

            file << std::endl;
            file.close();
        }
    }
};

} //end of dbn namespace

#endif