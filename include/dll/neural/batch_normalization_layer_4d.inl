//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/neural_layer.hpp"

namespace dll {

/*!
 * \brief Batch Normalization layer
 */
template <typename Desc>
struct batch_normalization_4d_layer : neural_layer<batch_normalization_4d_layer<Desc>, Desc> {
    using desc      = Desc;                                                   ///< The descriptor type
    using base_type = neural_layer<batch_normalization_4d_layer<Desc>, Desc>; ///< The base type
    using weight    = typename desc::weight;                                  ///< The data type of the layer

    static constexpr size_t Kernels = desc::Kernels; ///< The number of feature maps
    static constexpr size_t W       = desc::Width;   ///< The width of feature maps
    static constexpr size_t H       = desc::Height;  ///< The height of feature maps
    static constexpr weight e        = 1e-8;          ///< Epsilon for numerical stability

    etl::fast_matrix<weight, Kernels> gamma;
    etl::fast_matrix<weight, Kernels> beta;

    etl::fast_matrix<weight, Kernels> mean;
    etl::fast_matrix<weight, Kernels> var;

    etl::fast_matrix<weight, Kernels> last_mean;
    etl::fast_matrix<weight, Kernels> last_var;
    etl::fast_matrix<weight, Kernels> inv_var;

    etl::dyn_matrix<weight, 4> input_pre; /// B x K x W x H

    weight momentum = 0.9;

    // For SGD
    etl::fast_matrix<weight, Kernels>& w = gamma;
    etl::fast_matrix<weight, Kernels>& b = beta;

    //Backup gamma and beta
    std::unique_ptr<etl::fast_matrix<weight, Kernels>> bak_gamma; ///< Backup gamma
    std::unique_ptr<etl::fast_matrix<weight, Kernels>> bak_beta;  ///< Backup beta

    batch_normalization_4d_layer() : base_type() {
        gamma = 1.0;
        beta = 0.0;
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "batch_norm";
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    static constexpr size_t parameters() noexcept {
        return 4 * Kernels;
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    static constexpr size_t input_size() noexcept {
        return Kernels * W * H;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    static constexpr size_t output_size() noexcept {
        return Kernels * W * H;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void batch_activate_hidden(Output& output, const Input& input) const {
        test_batch_activate_hidden(output, input);
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void test_batch_activate_hidden(Output& output, const Input& input) const {
        const auto B = etl::dim<0>(input);

        auto inv_var = etl::force_temporary(1.0 / etl::sqrt(var + e));

        for (size_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < Kernels; ++k) {
                output(b)(k) = (gamma(k) >> ((input(b)(k) - mean(k)) >> inv_var(k))) + beta(k);
            }
        }
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void train_batch_activate_hidden(Output& output, const Input& input) {
        cpp_unused(output);

        const auto B = etl::dim<0>(input);
        const auto S = B * W * H;

        // Compute the mean of the mini-batch
        last_mean = etl::bias_batch_mean_4d(input);

        // Compute the variance of the mini-batch
        last_var  = 0;

        for (size_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < Kernels; ++k) {
                last_var(k) += etl::sum((input(b)(k) - last_mean(k)) >> (input(b)(k) - last_mean(k)));
            }
        }

        last_var /= S;

        inv_var  = 1.0 / etl::sqrt(last_var + e);

        input_pre.inherit_if_null(input);

        for(size_t b = 0; b < B; ++b){
            for (size_t k = 0; k < Kernels; ++k) {
                input_pre(b)(k) = (input(b)(k) - last_mean(k)) >> inv_var(k);
                output(b)(k)    = (gamma(k) >> input_pre(b)(k)) + beta(k);
            }
        }

        //// Update the current mean and variance
        mean = momentum * mean + (1.0 - momentum) * last_mean;
        var  = momentum * var + (1.0 - momentum) * (S / (S - 1) * last_var);
    }

    /*!
     * \brief Adapt the errors, called before backpropagation of the errors.
     *
     * This must be used by layers that have both an activation fnction and a non-linearity.
     *
     * \param context the training context
     */
    template<typename C>
    void adapt_errors(C& context) const {
        cpp_unused(context);
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename HH, typename C>
    void backward_batch(HH&& output, C& context) const {
        const auto B = etl::dim<0>(context.input);
        const auto S = B * W * H;

        auto dxhat = etl::force_temporary_dim_only(context.errors);

        for(size_t b = 0; b < B; ++b){
            for (size_t k = 0; k < Kernels; ++k) {
                dxhat(b)(k) = context.errors(b)(k) >> gamma(k);
            }
        }

        auto dxhat_l      = etl::bias_batch_sum_4d(dxhat);
        auto dxhat_xhat_l = etl::bias_batch_sum_4d(dxhat >> input_pre);

        *dxhat_l;
        *dxhat_xhat_l;

        for(size_t b = 0; b < B; ++b){
            for (size_t k = 0; k < Kernels; ++k) {
                output(b)(k) = ((1.0 / S) * inv_var(k)) >> (S * dxhat(b)(k) - dxhat_l(k) - (input_pre(b)(k) >> dxhat_xhat_l(k)));
            }
        }
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        // Gradients of gamma
        context.w_grad = etl::bias_batch_sum_4d(input_pre >> context.errors);

        // Gradients of beta
        context.b_grad = etl::bias_batch_sum_4d(context.errors);
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that needs to be initialized
     */
    template<typename DLayer>
    static void dyn_init(DLayer& dyn){
        dyn.init_layer(Kernels, W, H);
    }
};

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<batch_normalization_4d_layer<Desc>> {
    static constexpr bool is_neural     = true; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false;  ///< Indicates if the layer is a transform layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for batch_normalization_4d_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, batch_normalization_4d_layer<Desc>, L> {
    using layer_t = batch_normalization_4d_layer<Desc>; ///< The current layer type
    using weight  = typename layer_t::weight;           ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, Desc::Kernels, Desc::W, Desc::H> input;  ///< A batch of input
    etl::fast_matrix<weight, batch_size, Desc::Kernels, Desc::W, Desc::H> output; ///< A batch of output
    etl::fast_matrix<weight, batch_size, Desc::Kernels, Desc::W, Desc::H> errors; ///< A batch of errors

    etl::fast_matrix<weight, Desc::Kernels> w_grad;
    etl::fast_matrix<weight, Desc::Kernels> b_grad;

    sgd_context(layer_t& /*layer*/){}
};

} //end of dll namespace
