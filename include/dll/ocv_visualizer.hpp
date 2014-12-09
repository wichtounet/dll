//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_OPENCV_RBM_VISUALIZER_HPP
#define DLL_OPENCV_RBM_VISUALIZER_HPP

#include "cpp_utils/stop_watch.hpp"

#include "rbm_traits.hpp"
#include "dbn_traits.hpp"

#ifndef DLL_DETAIL_ONLY
#include <opencv2/opencv.hpp>
#endif

//TODO This needs serious refactorings, there is too much duplicated code between the different specializations
//     Moreover, it would be better to get rid of the static members

namespace dll {

namespace detail {

struct shape {
    const std::size_t width;
    const std::size_t height;
    constexpr shape(std::size_t width, std::size_t height) : width(width), height(height) {}
};

constexpr inline std::size_t ct_mid(std::size_t a, std::size_t b){
    return (a+b) / 2;
}

constexpr inline std::size_t ct_pow(std::size_t a){
    return a*a;
}

#ifdef __clang__

constexpr std::size_t ct_sqrt(std::size_t res, std::size_t l, std::size_t r){
    if(l == r){
        return r;
    } else {
        const auto mid = (r + l) / 2;

        if(mid * mid >= res){
            return ct_sqrt(res, l, mid);
        } else {
            return ct_sqrt(res, mid + 1, r);
        }
    }
}

constexpr inline std::size_t ct_sqrt(const std::size_t res){
    return ct_sqrt(res, 1, res);
}

constexpr inline std::size_t best_height(const std::size_t total){
    const auto width = ct_sqrt(total);
    const auto square = total / width;

    if(width * square >= total){
        return square;
    } else {
        return square + 1;
    }
}

#else

constexpr inline std::size_t ct_sqrt(std::size_t res, std::size_t l, std::size_t r){
    return
        l == r ? r
        : ct_sqrt(res, ct_pow(
            ct_mid(r, l)) >= res ? l : ct_mid(r, l) + 1,
            ct_pow(ct_mid(r, l)) >= res ? ct_mid(r, l) : r);
}

constexpr inline std::size_t ct_sqrt(const std::size_t res){
    return ct_sqrt(res, 1, res);
}

constexpr inline std::size_t best_height(const std::size_t total){
    return (ct_sqrt(total) * (total / ct_sqrt(total))) >= total ? total / ct_sqrt(total) : ((total / ct_sqrt(total)) + 1);
}

#endif

constexpr inline std::size_t best_width(const std::size_t total){
    return ct_sqrt(total);
}

} //end of namespace detail

#ifndef DLL_DETAIL_ONLY

template<typename RBM>
struct base_ocv_rbm_visualizer {
    cpp::stop_watch<std::chrono::seconds> watch;

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

        cv::namedWindow("RBM Training", cv::WINDOW_NORMAL);

        refresh();
    }

    void training_end(const RBM&){
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        std::cout << "Press on any key to close the window..." << std::endl;
        cv::waitKey(0);
    }

    void refresh(){
        cv::imshow("RBM Training", buffer_image);
        cv::waitKey(30);
    }
};


//rbm_ocv_config is used instead of directly passing the parameters because
//adding non-type template parameters would break dll::watcher

template<std::size_t P = 20, bool S = true>
struct rbm_ocv_config {
    static constexpr const auto padding = P;
    static constexpr const auto scale = S;
};

template<typename RBM, typename C = rbm_ocv_config<>, typename Enable = void>
struct opencv_rbm_visualizer : base_ocv_rbm_visualizer<RBM> {
    using rbm_t = RBM;

    static constexpr const detail::shape filter_shape{
        detail::best_width(rbm_t::num_visible), detail::best_height(rbm_t::num_visible)};

    static constexpr const detail::shape tile_shape{
        detail::best_width(rbm_t::num_hidden), detail::best_height(rbm_t::num_hidden)};

    static constexpr const auto scale = C::scale;
    static constexpr const auto padding = C::padding;

    using base_type = base_ocv_rbm_visualizer<RBM>;
    using base_type::buffer_image;
    using base_type::refresh;

    opencv_rbm_visualizer() :
        base_type(
            filter_shape.width * tile_shape.width + (tile_shape.height + 1) * 1 + 2 * padding,
            filter_shape.height * tile_shape.height + (tile_shape.height + 1) * 1 + 2 * padding)
    {}

    void epoch_end(std::size_t epoch, const rbm_training_context& context, const RBM& rbm){
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
            context.reconstruction_error, context.free_energy, context.sparsity);

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image, "epoch " + std::to_string(epoch), cv::Point(10,12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        for(std::size_t hi = 0; hi < tile_shape.width; ++hi){
            for(std::size_t hj = 0; hj < tile_shape.height; ++hj){
                auto real_h = hi * tile_shape.height + hj;

                if(real_h >= rbm_t::num_hidden){
                    break;
                }

                typename RBM::weight min;
                typename RBM::weight max;

                if(scale){
                    min = etl::min(rbm.w);
                    max = etl::max(rbm.w);
                }

                for(std::size_t i = 0; i < filter_shape.width; ++i){
                    for(std::size_t j = 0; j < filter_shape.height; ++j){
                        auto real_v = i * filter_shape.height + j;

                        if(real_v >= rbm_t::num_visible){
                            break;
                        }

                        auto value = rbm.w(real_v, real_h);

                        if(scale){
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding+1+hi*(filter_shape.height+1)+i,
                            padding+1+hj*(filter_shape.width+1)+j) = value * 255;
                    }
                }
            }
        }

        refresh();
    }
};

template<typename RBM, typename C>
struct opencv_rbm_visualizer<RBM, C, std::enable_if_t<rbm_traits<RBM>::is_convolutional()>> : base_ocv_rbm_visualizer<RBM> {
    using rbm_t = RBM;

    static constexpr const detail::shape filter_shape{rbm_t::NW, rbm_t::NW};
    static constexpr const detail::shape tile_shape{detail::best_width(rbm_t::K), detail::best_height(rbm_t::K)};

    static constexpr const auto scale = C::scale;
    static constexpr const auto padding = C::padding;

    using base_type = base_ocv_rbm_visualizer<RBM>;
    using base_type::buffer_image;
    using base_type::refresh;

    opencv_rbm_visualizer() :
        base_type(
            filter_shape.width * tile_shape.width + (tile_shape.height + 1) * 1 + 2 * padding,
            filter_shape.height * tile_shape.height + (tile_shape.height + 1) * 1 + 2 * padding)
    {}

    void epoch_end(std::size_t epoch, const rbm_training_context& context, const RBM& rbm){
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
            context.reconstruction_error, context.free_energy, context.sparsity);

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image, "epoch " + std::to_string(epoch), cv::Point(10,12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        std::size_t channel = 0;

        for(std::size_t hi = 0; hi < tile_shape.width; ++hi){
            for(std::size_t hj = 0; hj < tile_shape.height; ++hj){
                auto real_k = hi * tile_shape.height + hj;

                if(real_k >= rbm_t::K){
                    break;
                }

                typename RBM::weight min;
                typename RBM::weight max;

                if(scale){
                    min = etl::min(rbm.w(channel)(real_k));
                    max = etl::max(rbm.w(channel)(real_k));
                }

                for(std::size_t fi = 0; fi < filter_shape.width; ++fi){
                    for(std::size_t fj = 0; fj < filter_shape.height; ++fj){
                        auto value = rbm.w(channel, real_k, fi, fj);

                        if(scale){
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding+1+hi*(filter_shape.width+1)+fi,
                            padding+1+hj*(filter_shape.height+1)+fj) = value * 255;
                    }
                }
            }
        }

        refresh();
    }
};

template<typename DBN, typename C = rbm_ocv_config<>, typename Enable = void>
struct opencv_dbn_visualizer {
    static constexpr const bool ignore_sub = false;
    static constexpr const bool replace_sub = true;

    cpp::stop_watch<std::chrono::seconds> watch;

    using dbn_t = DBN;

    static std::vector<cv::Mat> buffer_images;
    static std::size_t current_image;

    opencv_dbn_visualizer() = default;

    //Pretraining phase

    void pretraining_begin(const DBN& /*dbn*/){
        std::cout << "DBN: Pretraining begin" << std::endl;

        cv::namedWindow("DBN Training", cv::WINDOW_NORMAL);
    }

    template<typename RBM>
    void pretrain_layer(const DBN& /*dbn*/, std::size_t I, std::size_t input_size){
        using rbm_t = RBM;
        static constexpr const auto num_visible = rbm_t::num_visible;
        static constexpr const auto num_hidden = rbm_t::num_hidden;

        std::cout << "DBN: Train layer " << I << " (" << num_visible << "->" << num_hidden << ") with " << input_size << " entries" << std::endl;

        current_image = I;
    }

    template<typename RBM>
    void training_begin(const RBM& rbm){
        using rbm_t = RBM;

        static constexpr const detail::shape filter_shape{
            detail::best_width(rbm_t::num_visible), detail::best_height(rbm_t::num_visible)};

        static constexpr const detail::shape tile_shape{
            detail::best_width(rbm_t::num_hidden), detail::best_height(rbm_t::num_hidden)};

        static constexpr const auto padding = C::padding;

        static constexpr const auto width = filter_shape.width * tile_shape.width + (tile_shape.width + 1) * 1 + 2 * padding;
        static constexpr const auto height = filter_shape.height * tile_shape.height + (tile_shape.height + 1) * 1 + 2 * padding;

        buffer_images.emplace_back(cv::Size(width, height), CV_8UC1);

        std::cout << "Train RBM with \"" << rbm_t::desc::template trainer_t<rbm_t>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << rbm.learning_rate << std::endl;

        if(rbm_traits<rbm_t>::has_momentum()){
            std::cout << "   momentum=" << rbm.momentum << std::endl;
        }

        if(w_decay(rbm_traits<RBM>::decay()) == decay_type::L1 || w_decay(rbm_traits<RBM>::decay()) == decay_type::L1L2){
            std::cout << "   weight_cost(L1)=" << rbm.l1_weight_cost << std::endl;
        }

        if(w_decay(rbm_traits<RBM>::decay()) == decay_type::L2 || w_decay(rbm_traits<RBM>::decay()) == decay_type::L1L2){
            std::cout << "   weight_cost(L2)=" << rbm.l2_weight_cost << std::endl;
        }

        if(rbm_traits<rbm_t>::has_sparsity()){
            std::cout << "   sparsity_target=" << rbm.sparsity_target << std::endl;
        }

        refresh();
    }

    template<typename RBM>
    void epoch_end(std::size_t epoch, const rbm_training_context& context, const RBM& rbm){
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
            context.reconstruction_error, context.free_energy, context.sparsity);

        using rbm_t = RBM;

        static constexpr const detail::shape filter_shape{
            detail::best_width(rbm_t::num_visible), detail::best_height(rbm_t::num_visible)};

        static constexpr const detail::shape tile_shape{
            detail::best_width(rbm_t::num_hidden), detail::best_height(rbm_t::num_hidden)};

        static constexpr const auto scale = C::scale;
        static constexpr const auto padding = C::padding;

        auto& buffer_image = buffer_images[current_image];

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image,
            "layer: " + std::to_string(current_image) + " epoch " + std::to_string(epoch),
            cv::Point(10,12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        for(std::size_t hi = 0; hi < tile_shape.width; ++hi){
            for(std::size_t hj = 0; hj < tile_shape.height; ++hj){
                auto real_h = hi * tile_shape.height + hj;

                if(real_h >= rbm_t::num_hidden){
                    break;
                }

                typename RBM::weight min;
                typename RBM::weight max;

                if(scale){
                    min = etl::min(rbm.w);
                    max = etl::max(rbm.w);
                }

                for(std::size_t i = 0; i < filter_shape.width; ++i){
                    for(std::size_t j = 0; j < filter_shape.height; ++j){
                        auto real_v = i * filter_shape.height + j;

                        if(real_v >= rbm_t::num_visible){
                            break;
                        }

                        auto value = rbm.w(real_v, real_h);

                        if(scale){
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding+1+hi*(filter_shape.width+1)+i,
                            padding+1+hj*(filter_shape.height+1)+j) = value * 255;
                    }
                }
            }
        }

        refresh();
    }

    template<typename RBM>
    void training_end(const RBM&){
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        std::cout << "Press on any key to close the window and continue training..." << std::endl;
        cv::waitKey(0);
    }

    void pretraining_end(const DBN& /*dbn*/){
        std::cout << "DBN: Pretraining end" << std::endl;
    }

    //Fine-tuning phase

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

        //TODO Would be interesting to update RBM images here
    }

    void fine_tuning_end(const DBN&){
        std::cout << "Total training took " << watch.elapsed() << "s" << std::endl;

        std::cout << "Press on any key to close the window" << std::endl;
        cv::waitKey(0);
    }

    //Utility functions

    void refresh(){
        cv::imshow("DBN Training", buffer_images[current_image]);
        cv::waitKey(30);
    }
};

template <typename DBN, typename C, typename Enable>
std::vector<cv::Mat> opencv_dbn_visualizer<DBN, C, Enable>::buffer_images;

template <typename DBN, typename C, typename Enable>
std::size_t opencv_dbn_visualizer<DBN, C, Enable>::current_image;

template<typename DBN, typename C>
struct opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_dynamic()>> {
    static constexpr const bool ignore_sub = false;
    static constexpr const bool replace_sub = true;

    cpp::stop_watch<std::chrono::seconds> watch;

    using dbn_t = DBN;

    static std::vector<cv::Mat> buffer_images;
    static std::size_t current_image;

    opencv_dbn_visualizer() = default;

    //Pretraining phase

    void pretraining_begin(const DBN& /*dbn*/){
        std::cout << "DBN: Pretraining begin" << std::endl;

        cv::namedWindow("DBN Training", cv::WINDOW_NORMAL);
    }

    template<typename RBM>
    void pretrain_layer(const DBN& /*dbn*/, std::size_t I, std::size_t input_size){
        printf("DBN: Train layer %lu with %lu entries", I, input_size);
        current_image = I;
    }

    template<typename RBM>
    void training_begin(const RBM& rbm){
        using rbm_t = RBM;

        auto visible = input_size(rbm);
        auto hidden = output_size(rbm);

        const detail::shape filter_shape{detail::best_width(visible), detail::best_height(visible)};
        const detail::shape tile_shape{detail::best_width(hidden), detail::best_height(hidden)};

        const auto padding = C::padding;

        const auto width = filter_shape.width * tile_shape.width + (tile_shape.width + 1) * 1 + 2 * padding;
        const auto height = filter_shape.height * tile_shape.height + (tile_shape.height + 1) * 1 + 2 * padding;

        buffer_images.emplace_back(cv::Size(width, height), CV_8UC1);

        std::cout << "Train RBM with \"" << rbm_t::desc::template trainer_t<rbm_t>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << rbm.learning_rate << std::endl;

        if(rbm_traits<rbm_t>::has_momentum()){
            std::cout << "   momentum=" << rbm.momentum << std::endl;
        }

        if(w_decay(rbm_traits<RBM>::decay()) == decay_type::L1 || w_decay(rbm_traits<RBM>::decay()) == decay_type::L1L2){
            std::cout << "   weight_cost(L1)=" << rbm.l1_weight_cost << std::endl;
        }

        if(w_decay(rbm_traits<RBM>::decay()) == decay_type::L2 || w_decay(rbm_traits<RBM>::decay()) == decay_type::L1L2){
            std::cout << "   weight_cost(L2)=" << rbm.l2_weight_cost << std::endl;
        }

        if(rbm_traits<rbm_t>::has_sparsity()){
            std::cout << "   sparsity_target=" << rbm.sparsity_target << std::endl;
        }

        refresh();
    }

    template<typename RBM>
    void epoch_end(std::size_t epoch, const rbm_training_context& context, const RBM& rbm){
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
            context.reconstruction_error, context.free_energy, context.sparsity);

        auto visible = input_size(rbm);
        auto hidden = output_size(rbm);

        const detail::shape filter_shape{detail::best_width(visible), detail::best_height(visible)};
        const detail::shape tile_shape{detail::best_width(hidden), detail::best_height(hidden)};

        static constexpr const auto scale = C::scale;
        static constexpr const auto padding = C::padding;

        auto& buffer_image = buffer_images[current_image];

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image,
            "layer: " + std::to_string(current_image) + " epoch " + std::to_string(epoch),
            cv::Point(10,12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        for(std::size_t hi = 0; hi < tile_shape.width; ++hi){
            for(std::size_t hj = 0; hj < tile_shape.height; ++hj){
                auto real_h = hi * tile_shape.height + hj;

                if(real_h >= hidden){
                    break;
                }

                typename RBM::weight min;
                typename RBM::weight max;

                if(scale){
                    min = etl::min(rbm.w);
                    max = etl::max(rbm.w);
                }

                for(std::size_t i = 0; i < filter_shape.width; ++i){
                    for(std::size_t j = 0; j < filter_shape.height; ++j){
                        auto real_v = i * filter_shape.height + j;

                        if(real_v >= visible){
                            break;
                        }

                        auto value = rbm.w(real_v, real_h);

                        if(scale){
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding+1+hi*(filter_shape.width+1)+i,
                            padding+1+hj*(filter_shape.height+1)+j) = value * 255;
                    }
                }
            }
        }

        refresh();
    }

    template<typename RBM>
    void training_end(const RBM&){
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        std::cout << "Press on any key to close the window and continue training..." << std::endl;
        cv::waitKey(0);
    }

    void pretraining_end(const DBN& /*dbn*/){
        std::cout << "DBN: Pretraining end" << std::endl;
    }

    //Utility functions

    void refresh(){
        cv::imshow("DBN Training", buffer_images[current_image]);
        cv::waitKey(30);
    }
};

template <typename DBN, typename C>
std::vector<cv::Mat> opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_dynamic()>>::buffer_images;

template <typename DBN, typename C>
std::size_t opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_dynamic()>>::current_image;

template<typename DBN, typename C>
struct opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_convolutional()>> {
    static constexpr const bool ignore_sub = false;
    static constexpr const bool replace_sub = true;

    cpp::stop_watch<std::chrono::seconds> watch;

    using dbn_t = DBN;

    static std::vector<cv::Mat> buffer_images;
    static std::size_t current_image;

    opencv_dbn_visualizer() = default;

    //Pretraining phase

    void pretraining_begin(const DBN& /*dbn*/){
        std::cout << "CDBN: Pretraining begin" << std::endl;

        cv::namedWindow("CDBN Training", cv::WINDOW_NORMAL);
    }

    template<typename RBM>
    void pretrain_layer(const DBN& /*dbn*/, std::size_t I, std::size_t input_size){
        using rbm_t = RBM;

        static constexpr const auto NC = rbm_t::NC;
        static constexpr const auto NV = rbm_t::NV;
        static constexpr const auto NH = rbm_t::NH;
        static constexpr const auto NW = rbm_t::NW;
        static constexpr const auto K = rbm_t::K;

        printf("CDBN: Train layer %lu (%lux%lux%lu -> %lux%lu -> %lux%lux%lu) with %lu entries \n", I, NV, NV, NC, NW, NW, NH, NH, K, input_size);

        current_image = I;
    }

    template<typename RBM>
    void training_begin(const RBM& rbm){
        using rbm_t = RBM;

        static constexpr const detail::shape filter_shape{rbm_t::NW, rbm_t::NW};
        static constexpr const detail::shape tile_shape{detail::best_width(rbm_t::K), detail::best_height(rbm_t::K)};

        static constexpr const auto padding = C::padding;

        static constexpr const auto width = filter_shape.width * tile_shape.width + (tile_shape.width + 1) * 1 + 2 * padding;
        static constexpr const auto height = filter_shape.height * tile_shape.height + (tile_shape.height + 1) * 1 + 2 * padding;

        std::cout <<cv::Size(width, height) << std::endl;

        buffer_images.emplace_back(cv::Size(width, height), CV_8UC1);

        std::cout << "Train RBM with \"" << rbm_t::desc::template trainer_t<rbm_t>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << rbm.learning_rate << std::endl;

        if(rbm_traits<rbm_t>::has_momentum()){
            std::cout << "   momentum=" << rbm.momentum << std::endl;
        }

        if(w_decay(rbm_traits<RBM>::decay()) == decay_type::L1 || w_decay(rbm_traits<RBM>::decay()) == decay_type::L1L2){
            std::cout << "   weight_cost(L1)=" << rbm.l1_weight_cost << std::endl;
        }

        if(w_decay(rbm_traits<RBM>::decay()) == decay_type::L2 || w_decay(rbm_traits<RBM>::decay()) == decay_type::L1L2){
            std::cout << "   weight_cost(L2)=" << rbm.l2_weight_cost << std::endl;
        }

        if(rbm_traits<rbm_t>::has_sparsity()){
            std::cout << "   sparsity_target=" << rbm.sparsity_target << std::endl;
        }

        refresh();
    }

    template<typename RBM>
    void epoch_end(std::size_t epoch, const rbm_training_context& context, const RBM& rbm){
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
            context.reconstruction_error, context.free_energy, context.sparsity);

        using rbm_t = RBM;

        static constexpr const detail::shape filter_shape{rbm_t::NW, rbm_t::NW};
        static constexpr const detail::shape tile_shape{detail::best_width(rbm_t::K), detail::best_height(rbm_t::K)};

        static constexpr const auto scale = C::scale;
        static constexpr const auto padding = C::padding;

        auto& buffer_image = buffer_images[current_image];

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image,
            "layer: " + std::to_string(current_image) + " epoch " + std::to_string(epoch),
            cv::Point(10,12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        std::size_t channel = 0;

        for(std::size_t hi = 0; hi < tile_shape.width; ++hi){
            for(std::size_t hj = 0; hj < tile_shape.height; ++hj){
                auto real_k = hi * tile_shape.height + hj;

                if(real_k >= rbm_t::K){
                    break;
                }

                typename RBM::weight min;
                typename RBM::weight max;

                if(scale){
                    min = etl::min(rbm.w(channel)(real_k));
                    max = etl::max(rbm.w(channel)(real_k));
                }

                for(std::size_t fi = 0; fi < filter_shape.width; ++fi){
                    for(std::size_t fj = 0; fj < filter_shape.height; ++fj){
                        auto value = rbm.w(channel, real_k, fi, fj);

                        if(scale){
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding+1+hi*(filter_shape.width+1)+fi,
                            padding+1+hj*(filter_shape.height+1)+fj) = value * 255;
                    }
                }
            }
        }

        refresh();
    }

    template<typename RBM>
    void training_end(const RBM&){
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        std::cout << "Press on any key to close the window and continue training..." << std::endl;
        cv::waitKey(0);
    }

    void pretraining_end(const DBN& /*dbn*/){
        std::cout << "CDBN: Pretraining end" << std::endl;
    }

    //Utility functions

    void refresh(){
        cv::imshow("CDBN Training", buffer_images[current_image]);
        cv::waitKey(30);
    }
};

template <typename DBN, typename C>
std::vector<cv::Mat> opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_convolutional()>>::buffer_images;

template <typename DBN, typename C>
std::size_t opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_convolutional()>>::current_image;

#endif

} //end of dll namespace

#endif
