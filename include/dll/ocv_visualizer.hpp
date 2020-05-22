//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/stop_watch.hpp"

#include "layer_traits.hpp"
#include "dbn_traits.hpp"

#ifndef DLL_DETAIL_ONLY
#include <opencv2/opencv.hpp>
#endif

//TODO This needs serious refactorings, there is too much duplicated code between the different specializations
//     Moreover, it would be better to get rid of the static members

namespace dll {

namespace detail {

/*!
 * \brief Simple helper for compile-time shape.
 */
struct shape {
    const size_t width;  ///< The width of the window
    const size_t height; ///< The height of the window

    /*!
     * \brief construct a new shape
     */
    constexpr shape(size_t width, size_t height)
            : width(width), height(height) {}
};

/*!
 * \brief Compute the middle of two numbers, rounded
 */
constexpr inline size_t ct_mid(size_t a, size_t b) {
    return (a + b) / 2;
}

/*!
 * \brief Compute the power of two numbers
 */
constexpr inline size_t ct_pow(size_t a) {
    return a * a;
}

#ifdef __clang__

constexpr size_t ct_sqrt(size_t res, size_t l, size_t r) {
    if (l == r) {
        return r;
    } else {
        const auto mid = (r + l) / 2;

        if (mid * mid >= res) {
            return ct_sqrt(res, l, mid);
        } else {
            return ct_sqrt(res, mid + 1, r);
        }
    }
}

/*!
 * \brief Compute the square root of res
 */
constexpr inline size_t ct_sqrt(const size_t res) {
    return ct_sqrt(res, 1, res);
}

/*!
 * \brief Compute the best height for the total number of weights
 */
constexpr inline size_t best_height(const size_t total) {
    const auto width  = ct_sqrt(total);
    const auto square = total / width;

    if (width * square >= total) {
        return square;
    } else {
        return square + 1;
    }
}

#else

constexpr inline size_t ct_sqrt(size_t res, size_t l, size_t r) {
    return l == r ? r
                  : ct_sqrt(res, ct_pow(
                                     ct_mid(r, l)) >= res
                                     ? l
                                     : ct_mid(r, l) + 1,
                            ct_pow(ct_mid(r, l)) >= res ? ct_mid(r, l) : r);
}

/*!
 * \brief Compute the square root of res
 */
constexpr inline size_t ct_sqrt(const size_t res) {
    return ct_sqrt(res, 1, res);
}

/*!
 * \brief Compute the best height for the total number of weights
 */
constexpr inline size_t best_height(const size_t total) {
    return (ct_sqrt(total) * (total / ct_sqrt(total))) >= total ? total / ct_sqrt(total) : ((total / ct_sqrt(total)) + 1);
}

#endif

/*!
 * \brief Compute the best width for the total number of weights
 */
constexpr inline size_t best_width(const size_t total) {
    return ct_sqrt(total);
}

} //end of namespace detail

#ifndef DLL_DETAIL_ONLY

/*!
 * \brief The base type for an OpenCV visualizer
 */
template <typename RBM>
struct base_ocv_rbm_visualizer {
    cpp::stop_watch<std::chrono::seconds> watch; ///< The timer for entire training

    const size_t width;  ///< The width of the view
    const size_t height; ///< The height of the view

    cv::Mat buffer_image; ///< The OpenCV buffer image

    /*!
     * \brief Initialize the base_ocv_rbm_visualizer
     */
    base_ocv_rbm_visualizer(size_t width, size_t height)
            : width(width), height(height), buffer_image(cv::Size(width, height), CV_8UC1) {
        //Nothing to init
    }

    /*!
     * \brief Indicates that the training has begin for the given RBM
     * \param rbm The RBM started training
     */
    void training_begin(const RBM& rbm) {
        std::cout << "Train RBM with \"" << RBM::desc::template trainer_t<RBM>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << rbm.learning_rate << std::endl;

        if (rbm_layer_traits<RBM>::has_momentum()) {
            std::cout << "   momentum=" << rbm.momentum << std::endl;
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

        cv::namedWindow("RBM Training", cv::WINDOW_NORMAL);

        refresh();
    }

    /*!
     * \brief Indicates that the training has finished for the given RBM
     * \param rbm The RBM stopped training
     */
    void training_end([[maybe_unused]] const RBM& rbm) {
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        std::cout << "Press on any key to close the window..." << std::endl;
        cv::waitKey(0);
    }

    /*!
     * \brief Indicates the end of a pretraining batch
     * \param rbm The RBM stopped training
     * \param context The training context
     * \param batch The batch that ended
     * \param batches The total number of batches
     */
    void batch_end([[maybe_unused]] const RBM& rbm, const rbm_training_context& context, size_t batch, size_t batches) {
        printf("Batch %ld/%ld - Reconstruction error: %.5f - Sparsity: %.5f\n", batch, batches,
               context.batch_error, context.batch_sparsity);
    }

    /*!
     * \brief Refresh the view
     */
    void refresh() {
        cv::imshow("RBM Training", buffer_image);
        cv::waitKey(30);
    }
};

//rbm_ocv_config is used instead of directly passing the parameters because
//adding non-type template parameters would break dll::watcher

template <size_t P = 20, bool S = true>
struct rbm_ocv_config {
    static constexpr auto padding = P; ///< The padding
    static constexpr auto scale   = S; ///< The scaling
};

template <typename RBM, typename C = rbm_ocv_config<>, typename Enable = void>
struct opencv_rbm_visualizer : base_ocv_rbm_visualizer<RBM> {
    using rbm_t = RBM; ///< The type of the RBM

    /*!
     * \brief The shape of a filter
     */
    static constexpr detail::shape filter_shape{
        detail::best_width(rbm_t::num_visible), detail::best_height(rbm_t::num_visible)};

    /*!
     * \brief The shape of a tile
     */
    static constexpr detail::shape tile_shape{
        detail::best_width(rbm_t::num_hidden), detail::best_height(rbm_t::num_hidden)};

    static constexpr auto scale   = C::scale;   ///< The scale
    static constexpr auto padding = C::padding; ///< The padding

    using base_type = base_ocv_rbm_visualizer<RBM>;
    using base_type::buffer_image;
    using base_type::refresh;

    opencv_rbm_visualizer()
            : base_type(
                  filter_shape.width * tile_shape.width + (tile_shape.height + 1) * 1 + 2 * padding,
                  filter_shape.height * tile_shape.height + (tile_shape.height + 1) * 1 + 2 * padding) {}

    void draw_weights(const RBM& rbm) {
        for (size_t hi = 0; hi < tile_shape.width; ++hi) {
            for (size_t hj = 0; hj < tile_shape.height; ++hj) {
                auto real_h = hi * tile_shape.height + hj;

                if (real_h >= rbm_t::num_hidden) {
                    break;
                }

                typename RBM::weight min;
                typename RBM::weight max;

                if (scale) {
                    min = etl::min(rbm.w);
                    max = etl::max(rbm.w);
                }

                for (size_t i = 0; i < filter_shape.width; ++i) {
                    for (size_t j = 0; j < filter_shape.height; ++j) {
                        auto real_v = i * filter_shape.height + j;

                        if (real_v >= rbm_t::num_visible) {
                            break;
                        }

                        auto value = rbm.w(real_v, real_h);

                        if (scale) {
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding + 1 + hi * (filter_shape.height + 1) + i,
                            padding + 1 + hj * (filter_shape.width + 1) + j) = value * 255;
                    }
                }
            }
        }
    }

    void epoch_start([[maybe_unused]] size_t epoch) {}

    /*!
     * \brief Indicates the end of an epoch of pretraining.
     * \param epoch The epoch that just finished training
     * \param context The RBM's training context
     * \param rbm The RBM being trained
     */
    void epoch_end(size_t epoch, const rbm_training_context& context, const RBM& rbm) {
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
               context.reconstruction_error, context.free_energy, context.sparsity);

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image, "epoch " + std::to_string(epoch), cv::Point(10, 12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        draw_weights(rbm);

        refresh();
    }
};

template <typename RBM, typename C>
struct opencv_rbm_visualizer<RBM, C, std::enable_if_t<layer_traits<RBM>::is_convolutional_rbm_layer()>> : base_ocv_rbm_visualizer<RBM> {
    using rbm_t = RBM; ///< The type of the RBM

    /*!
     * \brief The shape of a filter
     */
    static constexpr detail::shape filter_shape{rbm_t::NW1, rbm_t::NW2};

    /*!
     * \brief The shape of a tile
     */
    static constexpr detail::shape tile_shape{detail::best_width(rbm_t::K), detail::best_height(rbm_t::K)};

    static constexpr auto scale   = C::scale;   ///< The scale
    static constexpr auto padding = C::padding; ///< The padding

    using base_type = base_ocv_rbm_visualizer<RBM>;
    using base_type::buffer_image;
    using base_type::refresh;

    opencv_rbm_visualizer()
            : base_type(
                  filter_shape.width * tile_shape.width + (tile_shape.height + 1) * 1 + 2 * padding,
                  filter_shape.height * tile_shape.height + (tile_shape.height + 1) * 1 + 2 * padding) {}

    void draw_weights(const RBM& rbm) {
        size_t channel = 0;

        for (size_t hi = 0; hi < tile_shape.width; ++hi) {
            for (size_t hj = 0; hj < tile_shape.height; ++hj) {
                auto real_k = hi * tile_shape.height + hj;

                if (real_k >= rbm_t::K) {
                    break;
                }

                typename RBM::weight min;
                typename RBM::weight max;

                if (scale) {
                    min = etl::min(rbm.w(channel)(real_k));
                    max = etl::max(rbm.w(channel)(real_k));
                }

                for (size_t fi = 0; fi < filter_shape.width; ++fi) {
                    for (size_t fj = 0; fj < filter_shape.height; ++fj) {
                        auto value = rbm.w(channel, real_k, fi, fj);

                        if (scale) {
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding + 1 + hi * (filter_shape.width + 1) + fi,
                            padding + 1 + hj * (filter_shape.height + 1) + fj) = value * 255;
                    }
                }
            }
        }
    }

    void epoch_start([[maybe_unused]] size_t epoch) {}

    /*!
     * \brief Indicates the end of an epoch of pretraining.
     * \param epoch The epoch that just finished training
     * \param context The RBM's training context
     * \param rbm The RBM being trained
     */
    void epoch_end(size_t epoch, const rbm_training_context& context, const RBM& rbm) {
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
               context.reconstruction_error, context.free_energy, context.sparsity);

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image, "epoch " + std::to_string(epoch), cv::Point(10, 12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        draw_weights(rbm);

        refresh();
    }
};

template <typename DBN, typename C = rbm_ocv_config<>, typename Enable = void>
struct opencv_dbn_visualizer {
    static constexpr bool ignore_sub  = false; ///< For pretraining of a DBN, indicates if the regular RBM watcher should be used (false) or ignored (true)
    static constexpr bool replace_sub = true; ///< For pretraining of a DBN, indicates if the DBN watcher should replace (true) the RBM watcher or not (false)

    cpp::stop_watch<std::chrono::seconds> watch; ///< The timer for entire training

    using dbn_t = DBN; ///< The DBN being trained

    static std::vector<cv::Mat> buffer_images; ///< The buffer images
    static size_t current_image;               ///< The current images

    opencv_dbn_visualizer() = default;

    //Pretraining phase

    /*!
     * \brief The pretraining is beginning
     * \param dbn The DBN to pretrain
     * \param max_epochs The total number of epochs
     */
    void pretraining_begin([[maybe_unused]] const DBN& dbn, size_t max_epochs) {
        std::cout << "DBN: Pretraining begin for " << max_epochs << " epochs" << std::endl;

        cv::namedWindow("DBN Training", cv::WINDOW_NORMAL);
    }

    /*!
     * \brief Indicates that the given layer is starting pretraining
     * \param dbn The DBN being trained
     * \param I The index of the layer being pretraining
     * \param rbm The RBM being trained
     * \param input_size the number of inputs
     */
    template <typename RBM>
    void pretrain_layer(const DBN& dbn, size_t I, size_t input_size) {
        using rbm_t = RBM;

        static constexpr auto NV = rbm_t::num_visible;
        static constexpr auto NH = rbm_t::num_hidden;

        if (input_size > 0) {
            std::cout << "DBN: Train layer " << I << " (" << NV << "->" << NH << ") with " << input_size << " entries" << std::endl;
        } else {
            std::cout << "DBN: Train layer " << I << " (" << NV << "->" << NH << ")" << std::endl;
        }

        current_image = I;
    }

    /*!
     * \brief Indicates that the training has begin for the given RBM
     * \param rbm The RBM started training
     */
    template <typename RBM>
    void training_begin(const RBM& rbm) {
        using rbm_t = RBM;

        static constexpr detail::shape filter_shape{
            detail::best_width(rbm_t::num_visible), detail::best_height(rbm_t::num_visible)};

        static constexpr detail::shape tile_shape{
            detail::best_width(rbm_t::num_hidden), detail::best_height(rbm_t::num_hidden)};

        static constexpr auto padding = C::padding;

        static constexpr auto width  = filter_shape.width * tile_shape.width + (tile_shape.width + 1) * 1 + 2 * padding;
        static constexpr auto height = filter_shape.height * tile_shape.height + (tile_shape.height + 1) * 1 + 2 * padding;

        buffer_images.emplace_back(cv::Size(width, height), CV_8UC1);

        std::cout << "Train RBM with \"" << rbm_t::desc::template trainer_t<rbm_t, false>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << rbm.learning_rate << std::endl;

        if (rbm_layer_traits<rbm_t>::has_momentum()) {
            std::cout << "   momentum=" << rbm.momentum << std::endl;
        }

        if (w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1 || w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1L2) {
            std::cout << "   weight_cost(L1)=" << rbm.l1_weight_cost << std::endl;
        }

        if (w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L2 || w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1L2) {
            std::cout << "   weight_cost(L2)=" << rbm.l2_weight_cost << std::endl;
        }

        if (rbm_layer_traits<rbm_t>::has_sparsity()) {
            std::cout << "   sparsity_target=" << rbm.sparsity_target << std::endl;
        }

        refresh();
    }

    void epoch_start([[maybe_unused]] size_t epoch) {}

    /*!
     * \brief Indicates the end of an epoch of pretraining.
     * \param epoch The epoch that just finished training
     * \param context The RBM's training context
     * \param rbm The RBM being trained
     */
    template <typename RBM>
    void epoch_end(size_t epoch, const rbm_training_context& context, const RBM& rbm) {
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
               context.reconstruction_error, context.free_energy, context.sparsity);

        using rbm_t = RBM;

        static constexpr detail::shape filter_shape{
            detail::best_width(rbm_t::num_visible), detail::best_height(rbm_t::num_visible)};

        static constexpr detail::shape tile_shape{
            detail::best_width(rbm_t::num_hidden), detail::best_height(rbm_t::num_hidden)};

        static constexpr auto scale   = C::scale;
        static constexpr auto padding = C::padding;

        auto& buffer_image = buffer_images[current_image];

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image,
                    "layer: " + std::to_string(current_image) + " epoch " + std::to_string(epoch),
                    cv::Point(10, 12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        for (size_t hi = 0; hi < tile_shape.width; ++hi) {
            for (size_t hj = 0; hj < tile_shape.height; ++hj) {
                auto real_h = hi * tile_shape.height + hj;

                if (real_h >= rbm_t::num_hidden) {
                    break;
                }

                typename RBM::weight min;
                typename RBM::weight max;

                if (scale) {
                    min = etl::min(rbm.w);
                    max = etl::max(rbm.w);
                }

                for (size_t i = 0; i < filter_shape.width; ++i) {
                    for (size_t j = 0; j < filter_shape.height; ++j) {
                        auto real_v = i * filter_shape.height + j;

                        if (real_v >= rbm_t::num_visible) {
                            break;
                        }

                        auto value = rbm.w(real_v, real_h);

                        if (scale) {
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding + 1 + hi * (filter_shape.width + 1) + i,
                            padding + 1 + hj * (filter_shape.height + 1) + j) = value * 255;
                    }
                }
            }
        }

        refresh();
    }

    /*!
     * \brief Indicates the end of a pretraining batch
     * \param rbm The RBM stopped training
     * \param context The training context
     * \param batch The batch that ended
     * \param batches The total number of batches
     */
    template <typename RBM>
    void batch_end(const RBM& /* rbm */, const rbm_training_context& context, size_t batch, size_t batches) {
        printf("Batch %ld/%ld - Reconstruction error: %.5f - Sparsity: %.5f\n", batch, batches,
               context.batch_error, context.batch_sparsity);
    }

    /*!
     * \brief Indicates that the training has finished for the given RBM
     * \param rbm The RBM stopped training
     */
    template <typename RBM>
    void training_end(const RBM&) {
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        std::cout << "Press on any key to close the window and continue training..." << std::endl;
        cv::waitKey(0);
    }

    /*!
     * \brief Pretraining ended for the given DBN
     */
    void pretraining_end(const DBN& /*dbn*/) {
        std::cout << "DBN: Pretraining end" << std::endl;
    }

    /*!
     * \brief Pretraining ended for the given batch for the given DBN
     */
    void pretraining_batch([[maybe_unused]] const DBN& dbn, size_t batch) {
        std::cout << "DBN: Pretraining batch " << batch << std::endl;
    }

    //Fine-tuning phase

    /*!
     * \brief Fine-tuning of the given network just started
     * \param dbn The DBN that is being trained
     * \param max_epochs The maximum number of epochs to train the network
     */
    void fine_tuning_begin(const DBN& dbn) {
        std::cout << "Train DBN with \"" << DBN::desc::template trainer_t<DBN>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << dbn.learning_rate << std::endl;

        if (dbn_traits<DBN>::has_momentum()) {
            std::cout << "   momentum=" << dbn.momentum << std::endl;
        }
    }

    /*!
     * \brief One fine-tuning epoch is over
     * \param epoch The current epoch
     * \param error The current error
     * \param loss The current loss
     * \param dbn The network being trained
     */
    void ft_epoch_end(size_t epoch, double error, [[maybe_unused]] const DBN& dbn) {
        printf("epoch %ld - Classification error: %.5f \n", epoch, error);

        //TODO Would be interesting to update RBM images here
    }

    /*!
     * \brief Fine-tuning of the given network just finished
     * \param dbn The DBN that is being trained
     */
    void fine_tuning_end(const DBN&) {
        std::cout << "Total training took " << watch.elapsed() << "s" << std::endl;

        std::cout << "Press on any key to close the window" << std::endl;
        cv::waitKey(0);
    }

    //Utility functions

    /*!
     * \brief Refresh the view
     */
    void refresh() {
        cv::imshow("DBN Training", buffer_images[current_image]);
        cv::waitKey(30);
    }
};

template <typename DBN, typename C, typename Enable>
std::vector<cv::Mat> opencv_dbn_visualizer<DBN, C, Enable>::buffer_images;

template <typename DBN, typename C, typename Enable>
size_t opencv_dbn_visualizer<DBN, C, Enable>::current_image;

template <typename DBN, typename C>
struct opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_dynamic()>> {
    static constexpr bool ignore_sub  = false; ///< For pretraining of a DBN, indicates if the regular RBM watcher should be used (false) or ignored (true)
    static constexpr bool replace_sub = true; ///< For pretraining of a DBN, indicates if the DBN watcher should replace (true) the RBM watcher or not (false)

    cpp::stop_watch<std::chrono::seconds> watch; ///< The timer for entire training

    using dbn_t = DBN; ///< The type of the DBN

    static std::vector<cv::Mat> buffer_images; ///< The buffer images
    static size_t current_image;               ///< The current images

    opencv_dbn_visualizer() = default;

    //Pretraining phase

    /*!
     * \brief Indicates that the pretraining has begun for the given
     * DBN
     * \param dbn The DBN being pretrained
     * \param max_epochs The maximum number of epochs
     */
    void pretraining_begin([[maybe_unused]] const DBN& dbn, size_t max_epochs) {
        std::cout << "DBN: Pretraining begin for " << max_epochs << " epochs" << std::endl;

        cv::namedWindow("DBN Training", cv::WINDOW_NORMAL);
    }

    /*!
     * \brief Indicates that the given layer is starting pretraining
     * \param dbn The DBN being trained
     * \param I The index of the layer being pretraining
     * \param rbm The RBM being trained
     * \param input_size the number of inputs
     */
    template <typename RBM>
    void pretrain_layer([[maybe_unused]] const DBN& dbn, size_t I, size_t input_size) {
        printf("DBN: Train layer %lu with %lu entries\n", I, input_size);
        current_image = I;
    }

    /*!
     * \brief Indicates that the training has begin for the given RBM
     * \param rbm The RBM started training
     */
    template <typename RBM>
    void training_begin(const RBM& rbm) {
        using rbm_t = RBM;

        auto visible = input_size(rbm);
        auto hidden  = output_size(rbm);

        const detail::shape filter_shape{detail::best_width(visible), detail::best_height(visible)};
        const detail::shape tile_shape{detail::best_width(hidden), detail::best_height(hidden)};

        const auto padding = C::padding;

        const auto width  = filter_shape.width * tile_shape.width + (tile_shape.width + 1) * 1 + 2 * padding;
        const auto height = filter_shape.height * tile_shape.height + (tile_shape.height + 1) * 1 + 2 * padding;

        buffer_images.emplace_back(cv::Size(width, height), CV_8UC1);

        std::cout << "Train RBM with \"" << rbm_t::desc::template trainer_t<rbm_t, false>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << rbm.learning_rate << std::endl;

        if (rbm_layer_traits<rbm_t>::has_momentum()) {
            std::cout << "   momentum=" << rbm.momentum << std::endl;
        }

        if (w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1 || w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1L2) {
            std::cout << "   weight_cost(L1)=" << rbm.l1_weight_cost << std::endl;
        }

        if (w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L2 || w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1L2) {
            std::cout << "   weight_cost(L2)=" << rbm.l2_weight_cost << std::endl;
        }

        if (rbm_layer_traits<rbm_t>::has_sparsity()) {
            std::cout << "   sparsity_target=" << rbm.sparsity_target << std::endl;
        }

        refresh();
    }

    void epoch_start([[maybe_unused]] size_t epoch) {}

    template <typename RBM>
    void epoch_end(size_t epoch, const rbm_training_context& context, const RBM& rbm) {
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
               context.reconstruction_error, context.free_energy, context.sparsity);

        auto visible = input_size(rbm);
        auto hidden  = output_size(rbm);

        const detail::shape filter_shape{detail::best_width(visible), detail::best_height(visible)};
        const detail::shape tile_shape{detail::best_width(hidden), detail::best_height(hidden)};

        static constexpr auto scale   = C::scale;
        static constexpr auto padding = C::padding;

        auto& buffer_image = buffer_images[current_image];

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image,
                    "layer: " + std::to_string(current_image) + " epoch " + std::to_string(epoch),
                    cv::Point(10, 12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        for (size_t hi = 0; hi < tile_shape.width; ++hi) {
            for (size_t hj = 0; hj < tile_shape.height; ++hj) {
                auto real_h = hi * tile_shape.height + hj;

                if (real_h >= hidden) {
                    break;
                }

                typename RBM::weight min;
                typename RBM::weight max;

                if (scale) {
                    min = etl::min(rbm.w);
                    max = etl::max(rbm.w);
                }

                for (size_t i = 0; i < filter_shape.width; ++i) {
                    for (size_t j = 0; j < filter_shape.height; ++j) {
                        auto real_v = i * filter_shape.height + j;

                        if (real_v >= visible) {
                            break;
                        }

                        auto value = rbm.w(real_v, real_h);

                        if (scale) {
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding + 1 + hi * (filter_shape.width + 1) + i,
                            padding + 1 + hj * (filter_shape.height + 1) + j) = value * 255;
                    }
                }
            }
        }

        refresh();
    }

    /*!
     * \brief Indicates the end of a pretraining batch
     * \param rbm The RBM stopped training
     * \param context The training context
     * \param batch The batch that ended
     * \param batches The total number of batches
     */
    template <typename RBM>
    void batch_end(const RBM& /* rbm */, const rbm_training_context& context, size_t batch, size_t batches) {
        printf("Batch %ld/%ld - Reconstruction error: %.5f - Sparsity: %.5f\n", batch, batches,
               context.batch_error, context.batch_sparsity);
    }

    /*!
     * \brief Indicates that the training has finished for the given RBM
     * \param rbm The RBM stopped training
     */
    template <typename RBM>
    void training_end(const RBM&) {
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        std::cout << "Press on any key to close the window and continue training..." << std::endl;
        cv::waitKey(0);
    }

    /*!
     * \brief Pretraining ended for the given DBN
     */
    void pretraining_end([[maybe_unused]] const DBN& /*dbn*/) {
        std::cout << "DBN: Pretraining end" << std::endl;
    }

    /*!
     * \brief Pretraining ended for the given batch for the given DBN
     */
    void pretraining_batch([[maybe_unused]] const DBN& dbn, size_t batch) {
        std::cout << "DBN: Pretraining batch " << batch << std::endl;
    }

    //Utility functions

    /*!
     * \brief Refresh the view
     */
    void refresh() {
        cv::imshow("DBN Training", buffer_images[current_image]);
        cv::waitKey(30);
    }
};

template <typename DBN, typename C>
std::vector<cv::Mat> opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_dynamic()>>::buffer_images;

template <typename DBN, typename C>
size_t opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_dynamic()>>::current_image;

template <typename DBN, typename C>
struct opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_convolutional_rbm_layer()>> {
    static constexpr bool ignore_sub  = false; ///< For pretraining of a DBN, indicates if the regular RBM watcher should be used (false) or ignored (true)
    static constexpr bool replace_sub = true; ///< For pretraining of a DBN, indicates if the DBN watcher should replace (true) the RBM watcher or not (false)

    cpp::stop_watch<std::chrono::seconds> watch; ///< The timer for entire training

    using dbn_t = DBN; ///< The DBN type

    static std::vector<cv::Mat> buffer_images; ///< The buffer images
    static size_t current_image;               ///< The current images

    opencv_dbn_visualizer() = default;

    //Pretraining phase

    /*!
     * \brief Indicates that the pretraining has begun for the given
     * DBN
     * \param dbn The DBN being pretrained
     * \param max_epochs The maximum number of epochs
     */
    void pretraining_begin([[maybe_unused]] const DBN& dbn, size_t max_epochs) {
        std::cout << "CDBN: Pretraining begin for " << max_epochs << " epochs" << std::endl;

        cv::namedWindow("CDBN Training", cv::WINDOW_NORMAL);
    }

    template <typename RBM>
    void pretrain_layer([[maybe_unused]] const DBN& dbn, size_t I, size_t input_size) {
        using rbm_t = RBM;

        static constexpr auto NC  = rbm_t::NC;
        static constexpr auto NV1 = rbm_t::NV1;
        static constexpr auto NV2 = rbm_t::NV2;
        static constexpr auto NH1 = rbm_t::NH1;
        static constexpr auto NH2 = rbm_t::NH2;
        static constexpr auto NW1 = rbm_t::NW1;
        static constexpr auto NW2 = rbm_t::NW2;
        static constexpr auto K   = rbm_t::K;

        printf("CDBN: Train layer %lu (%lux%lux%lu -> %lux%lu -> %lux%lux%lu) with %lu entries \n", I, NV1, NV2, NC, NW1, NW2, NH1, NH2, K, input_size);

        current_image = I;
    }

    /*!
     * \brief Indicates that the training has begin for the given RBM
     * \param rbm The RBM started training
     */
    template <typename RBM>
    void training_begin(const RBM& rbm) {
        using rbm_t = RBM;

        static constexpr detail::shape filter_shape{rbm_t::NW1, rbm_t::NW2};
        static constexpr detail::shape tile_shape{detail::best_width(rbm_t::K), detail::best_height(rbm_t::K)};

        static constexpr auto padding = C::padding;

        static constexpr auto width  = filter_shape.width * tile_shape.width + (tile_shape.width + 1) * 1 + 2 * padding;
        static constexpr auto height = filter_shape.height * tile_shape.height + (tile_shape.height + 1) * 1 + 2 * padding;

        std::cout << cv::Size(width, height) << std::endl;

        buffer_images.emplace_back(cv::Size(width, height), CV_8UC1);

        std::cout << "Train RBM with \"" << rbm_t::desc::template trainer_t<rbm_t, false>::name() << "\"" << std::endl;
        std::cout << "With parameters:" << std::endl;
        std::cout << "   learning_rate=" << rbm.learning_rate << std::endl;

        if (rbm_layer_traits<rbm_t>::has_momentum()) {
            std::cout << "   momentum=" << rbm.momentum << std::endl;
        }

        if (w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1 || w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1L2) {
            std::cout << "   weight_cost(L1)=" << rbm.l1_weight_cost << std::endl;
        }

        if (w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L2 || w_decay(rbm_layer_traits<RBM>::decay()) == decay_type::L1L2) {
            std::cout << "   weight_cost(L2)=" << rbm.l2_weight_cost << std::endl;
        }

        if (rbm_layer_traits<rbm_t>::has_sparsity()) {
            std::cout << "   sparsity_target=" << rbm.sparsity_target << std::endl;
        }

        refresh();
    }

    void epoch_start([[maybe_unused]] size_t epoch) {}

    template <typename RBM>
    void epoch_end(size_t epoch, const rbm_training_context& context, const RBM& rbm) {
        printf("epoch %ld - Reconstruction error: %.5f - Free energy: %.3f - Sparsity: %.5f\n", epoch,
               context.reconstruction_error, context.free_energy, context.sparsity);

        using rbm_t = RBM;

        static constexpr detail::shape filter_shape{rbm_t::NW1, rbm_t::NW2};
        static constexpr detail::shape tile_shape{detail::best_width(rbm_t::K), detail::best_height(rbm_t::K)};

        static constexpr auto scale   = C::scale;
        static constexpr auto padding = C::padding;

        auto& buffer_image = buffer_images[current_image];

        buffer_image = cv::Scalar(255);

        cv::putText(buffer_image,
                    "layer: " + std::to_string(current_image) + " epoch " + std::to_string(epoch),
                    cv::Point(10, 12), CV_FONT_NORMAL, 0.3, cv::Scalar(0), 1, 2);

        size_t channel = 0;

        for (size_t hi = 0; hi < tile_shape.width; ++hi) {
            for (size_t hj = 0; hj < tile_shape.height; ++hj) {
                auto real_k = hi * tile_shape.height + hj;

                if (real_k >= rbm_t::K) {
                    break;
                }

                typename RBM::weight min;
                typename RBM::weight max;

                if (scale) {
                    min = etl::min(rbm.w(channel)(real_k));
                    max = etl::max(rbm.w(channel)(real_k));
                }

                for (size_t fi = 0; fi < filter_shape.width; ++fi) {
                    for (size_t fj = 0; fj < filter_shape.height; ++fj) {
                        auto value = rbm.w(channel, real_k, fi, fj);

                        if (scale) {
                            value -= min;
                            value *= 1.0 / (max + 1e-8);
                        }

                        buffer_image.template at<uint8_t>(
                            padding + 1 + hi * (filter_shape.width + 1) + fi,
                            padding + 1 + hj * (filter_shape.height + 1) + fj) = value * 255;
                    }
                }
            }
        }

        refresh();
    }

    /*!
     * \brief Indicates the end of a pretraining batch
     * \param rbm The RBM stopped training
     * \param context The training context
     * \param batch The batch that ended
     * \param batches The total number of batches
     */
    template <typename RBM>
    void batch_end(const RBM& /* rbm */, const rbm_training_context& context, size_t batch, size_t batches) {
        printf("Batch %ld/%ld - Reconstruction error: %.5f - Sparsity: %.5f\n", batch, batches,
               context.batch_error, context.batch_sparsity);
    }

    /*!
     * \brief Indicates that the training has finished for the given RBM
     * \param rbm The RBM stopped training
     */
    template <typename RBM>
    void training_end(const RBM&) {
        std::cout << "Training took " << watch.elapsed() << "s" << std::endl;

        std::cout << "Press on any key to close the window and continue training..." << std::endl;
        cv::waitKey(0);
    }

    /*!
     * \brief Pretraining ended for the given DBN
     */
    void pretraining_end(const DBN& /*dbn*/) {
        std::cout << "CDBN: Pretraining end" << std::endl;
    }

    //Utility functions

    void refresh() {
        cv::imshow("CDBN Training", buffer_images[current_image]);
        cv::waitKey(30);
    }
};

template <typename DBN, typename C>
std::vector<cv::Mat> opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_convolutional_rbm_layer()>>::buffer_images;

template <typename DBN, typename C>
size_t opencv_dbn_visualizer<DBN, C, std::enable_if_t<dbn_traits<DBN>::is_convolutional_rbm_layer()>>::current_image;

template <typename RBM>
void visualize_rbm(const RBM& rbm) {
    cv::namedWindow("RBM Training", cv::WINDOW_NORMAL);

    opencv_rbm_visualizer<RBM> visualizer;
    visualizer.draw_weights(rbm);
    visualizer.refresh();
    cv::waitKey(0);
}

#endif

} //end of dll namespace
