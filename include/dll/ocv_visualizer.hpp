//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_OPENCV_RBM_VISUALIZER_HPP
#define DLL_OPENCV_RBM_VISUALIZER_HPP

#include "dll/stop_watch.hpp"
#include "dll/rbm_traits.hpp"
#include "dll/dbn_traits.hpp"

#include <opencv2/opencv.hpp>

namespace dll {

template<typename RBM>
struct opencv_rbm_visualizer {
    stop_watch<std::chrono::seconds> watch;

    cv::Mat buffer_image;

    bool scale = true;
    std::size_t padding = 20;

    std::size_t shape = 28;
    std::size_t num_hidden = 10;

    std::size_t width = shape * num_hidden + (num_hidden + 1) * 1 + 2 * padding;
    std::size_t height = shape * num_hidden + (num_hidden + 1) * 1 + 2 * padding;

    opencv_rbm_visualizer() : buffer_image(cv::Size(width, height), CV_8UC1) {}

    void update_sizes(){
        width = shape * num_hidden + (num_hidden + 1) * 1 + 2 * padding;
        height = shape * num_hidden + (num_hidden + 1) * 1 + 2 * padding;
    }

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

        cv::namedWindow("RBM Training", cv::WINDOW_NORMAL);

        refresh();
    }

    void epoch_end(std::size_t epoch, double error, double free_energy, const RBM& rbm){
        printf("epoch %ld - Reconstruction error average: %.5f - Free energy average: %.3f\n", epoch, error, free_energy);

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image, "epoch " + std::to_string(epoch), cv::Point(10,12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        for(std::size_t hi = 0; hi < num_hidden; ++hi){
            for(std::size_t hj = 0; hj < num_hidden; ++hj){
                auto real_h = hi * num_hidden + hj;

                typename RBM::weight min = 100.0;
                typename RBM::weight max = 0.0;

                if(scale){
                    for(std::size_t real_v = 0; real_v < shape * shape; ++real_v){
                        min = std::min(rbm.w(real_v, real_h), min);
                        max = std::max(rbm.w(real_v, real_h), max);
                    }
                }

                for(std::size_t i = 0; i < shape; ++i){
                    for(std::size_t j = 0; j < shape; ++j){
                        auto real_v = i * shape + j;

                        auto value = rbm.w(real_v, real_h);

                        if(scale){
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.at<uint8_t>(padding+1+hi*(shape+1)+i, padding+1+hj*(shape+1)+j) = value * 255;
                    }
                }
            }
        }

        refresh();
    }

    void training_end(const RBM&){
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        cv::waitKey(0);
    }

    void refresh(){
        cv::imshow("RBM Training", buffer_image);
        cv::waitKey(30);
    }
};

} //end of dll namespace

#endif