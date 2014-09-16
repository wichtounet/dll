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

#include "stop_watch.hpp"
#include "rbm_traits.hpp"
#include "dbn_traits.hpp"

namespace dll {

template<typename RBM>
struct default_rbm_watcher {
    stop_watch<std::chrono::seconds> watch;

    void training_begin(const RBM& rbm){
        std::cout << "Train RBM with \"" << RBM::desc::template trainer_t<RBM>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << rbm.learning_rate << std::endl;

        if(rbm_traits<RBM>::has_momentum()){
            std::cout << "   momentum=" << rbm.momentum << std::endl;
        }

        if(rbm_traits<RBM>::decay() == decay_type::L1 || rbm_traits<RBM>::decay() == decay_type::L1_FULL){
            std::cout << "   weight_cost(L1)=" << rbm.weight_cost << std::endl;
        }

        if(rbm_traits<RBM>::decay() == decay_type::L2 || rbm_traits<RBM>::decay() == decay_type::L2_FULL){
            std::cout << "   weight_cost(L2)=" << rbm.weight_cost << std::endl;
        }

        if(rbm_traits<RBM>::has_sparsity()){
            std::cout << "   sparsity_target=" << rbm.sparsity_target << std::endl;
        }
    }

    void epoch_end(std::size_t epoch, double error, double free_energy, const RBM& /*rbm*/){
        printf("epoch %ld - Reconstruction error average: %.5f - Free energy average: %.3f\n", epoch, error, free_energy);
    }

    void training_end(const RBM&){
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;
    }
};

template<typename DBN>
struct default_dbn_watcher {
    static constexpr const bool ignore_sub = false;

    stop_watch<std::chrono::seconds> watch;

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
struct silent_dbn_watcher : default_dbn_watcher<DBN> {
    static constexpr const bool ignore_sub = true;
};

template<typename RBM>
struct histogram_watcher {
    default_rbm_watcher<RBM> parent;

    void training_begin(const RBM& rbm){
        parent.training_begin(rbm);
    }

    void epoch_end(std::size_t epoch, double error, double /*free_energy*/, const RBM& rbm){
        parent.epoch_end(epoch, error, rbm);
    }

    void training_end(const RBM& rbm){
        parent.training_end(rbm);
    }

    void generate_hidden_images(size_t epoch, const RBM& rbm){
        mkdir("reports", 0777);

        auto folder = "reports/epoch_" + std::to_string(epoch);
        mkdir(folder.c_str(), 0777);

        for(size_t j = 0; j < RBM::num_hidden; ++j){
            auto path = folder + "/h_" + std::to_string(j) + ".dat";
            std::ofstream file(path, std::ios::out);

            if(!file){
                std::cout << "Could not open file " << path << std::endl;
            } else {
                size_t i = RBM::num_visible;
                while(i > 0){
                    --i;

                    auto value = rbm.w(i,j);
                    file << static_cast<size_t>(value > 0 ? static_cast<size_t>(value * 255.0) << 8 : static_cast<size_t>(-value * 255.0) << 16) << " ";
                }

                file << std::endl;
                file.close();
            }
        }
    }

    void generate_histograms(size_t epoch, const RBM& rbm){
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