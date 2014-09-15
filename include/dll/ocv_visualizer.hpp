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
struct base_ocv_rbm_visualizer {
    stop_watch<std::chrono::seconds> watch;

    const std::size_t width;
    const std::size_t height;

    cv::Mat buffer_image;

    base_ocv_rbm_visualizer(std::size_t width, std::size_t height) :
            width(width), height(height),
            buffer_image(cv::Size(width, height), CV_8UC1) {
        //Nothing to init
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

    void training_end(const RBM&){
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        cv::waitKey(0);
    }

    void refresh(){
        cv::imshow("RBM Training", buffer_image);
        cv::waitKey(30);
    }
};

template<typename RBM, typename Enable = void>
struct opencv_rbm_visualizer : base_ocv_rbm_visualizer<RBM> {
    static constexpr const auto filter_shape = RBM::num_visible;

    const std::size_t num_hidden = 10;
    const bool scale = true;
    const std::size_t padding = 20;

    using base_type = base_ocv_rbm_visualizer<RBM>;
    using base_type::buffer_image;
    using base_type::refresh;

    opencv_rbm_visualizer(std::size_t num_hidden = 10, bool scale = true, std::size_t padding = 20) :
        base_type(
            filter_shape * num_hidden + (num_hidden + 1) * 1 + 2 * padding,
            filter_shape * num_hidden + (num_hidden + 1) * 1 + 2 * padding),
        num_hidden(num_hidden),
        scale(scale),
        padding(padding)
    {}

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
                    for(std::size_t real_v = 0; real_v < filter_shape * filter_shape; ++real_v){
                        min = std::min(rbm.w(real_v, real_h), min);
                        max = std::max(rbm.w(real_v, real_h), max);
                    }
                }

                for(std::size_t i = 0; i < filter_shape; ++i){
                    for(std::size_t j = 0; j < filter_shape; ++j){
                        auto real_v = i * filter_shape + j;

                        auto value = rbm.w(real_v, real_h);

                        if(scale){
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding+1+hi*(filter_shape+1)+i,
                            padding+1+hj*(filter_shape+1)+j) = value * 255;
                    }
                }
            }
        }

        refresh();
    }
};

template<typename RBM>
struct opencv_rbm_visualizer<RBM, enable_if_t<rbm_traits<RBM>::is_convolutional()>> : base_ocv_rbm_visualizer<RBM> {
    static constexpr const auto filter_shape = RBM::NW;

    const std::size_t num_hidden;

    const bool scale;
    const std::size_t padding;

    using base_type = base_ocv_rbm_visualizer<RBM>;
    using base_type::buffer_image;
    using base_type::refresh;

    opencv_rbm_visualizer(std::size_t num_hidden = 6, bool scale = true, std::size_t padding = 20) :
        base_type(
            filter_shape * num_hidden + (num_hidden + 1) * 1 + 2 * padding,
            filter_shape * num_hidden + (num_hidden + 1) * 1 + 2 * padding),
        num_hidden(num_hidden),
        scale(scale),
        padding(padding)
    {}

    void epoch_end(std::size_t epoch, double error, double free_energy, const RBM& rbm){
        printf("epoch %ld - Reconstruction error average: %.5f - Free energy average: %.3f\n", epoch, error, free_energy);

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image, "epoch " + std::to_string(epoch), cv::Point(10,12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        for(std::size_t hi = 0; hi < num_hidden; ++hi){
            for(std::size_t hj = 0; hj < num_hidden; ++hj){
                auto real_k = hi * num_hidden + hj;

                dll_assert(real_k < RBM::K, "Invalid filter index (>= K)");

                typename RBM::weight min = 100.0;
                typename RBM::weight max = 0.0;

                if(scale){
                    for(std::size_t fi = 0; fi < filter_shape; ++fi){
                        for(std::size_t fj = 0; fj < filter_shape; ++fj){
                            min = std::min(rbm.w(real_k)(fi, fj), min);
                            max = std::max(rbm.w(real_k)(fi, fj), max);
                        }
                    }
                }

                for(std::size_t fi = 0; fi < filter_shape; ++fi){
                    for(std::size_t fj = 0; fj < filter_shape; ++fj){
                        auto value = rbm.w(real_k)(fi, fj);

                        if(scale){
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding+1+hi*(filter_shape+1)+fi,
                            padding+1+hj*(filter_shape+1)+fj) = value * 255;
                    }
                }
            }
        }

        refresh();
    }
};

} //end of dll namespace

#endif