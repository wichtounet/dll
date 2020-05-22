//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
struct dyn_batch_normalization_4d_layer_impl : neural_layer<dyn_batch_normalization_4d_layer_impl<Desc>, Desc> {
    using desc        = Desc;                                                            ///< The descriptor type
    using base_type   = neural_layer<dyn_batch_normalization_4d_layer_impl<Desc>, Desc>; ///< The base type
    using weight      = typename desc::weight;                                           ///< The data type of the layer
    using this_type   = dyn_batch_normalization_4d_layer_impl<Desc>;                     ///< The type of this layer
    using layer_t     = this_type;                                                       ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;                                      ///< The dynamic version of this layer

    static constexpr weight e = 1e-8; ///< Epsilon for numerical stability

    using input_one_t  = etl::dyn_matrix<weight, 3>; ///< The type of one input
    using output_one_t = etl::dyn_matrix<weight, 3>; ///< The type of one output
    using input_t      = std::vector<input_one_t>;   ///< The type of the input
    using output_t     = std::vector<output_one_t>;  ///< The type of the output

    etl::dyn_matrix<weight, 1> gamma;
    etl::dyn_matrix<weight, 1> beta;

    etl::dyn_matrix<weight, 1> mean;
    etl::dyn_matrix<weight, 1> var;

    etl::dyn_matrix<weight, 1> last_mean;
    etl::dyn_matrix<weight, 1> last_var;
    etl::dyn_matrix<weight, 1> inv_var;

    etl::dyn_matrix<weight, 4> input_pre; /// B x K x W x H

    weight momentum = 0.9;

    //Backup gamma and beta
    std::unique_ptr<etl::dyn_matrix<weight, 1>> bak_gamma; ///< Backup gamma
    std::unique_ptr<etl::dyn_matrix<weight, 1>> bak_beta;  ///< Backup beta

    size_t Kernels = 0;
    size_t W       = 0;
    size_t H       = 0;

    dyn_batch_normalization_4d_layer_impl() : base_type() {
        // Nothing else to init
    }

    void init_layer(size_t Kernels, size_t W, size_t H){
        this->Kernels = Kernels;
        this->W       = W;
        this->H       = H;

        gamma = etl::dyn_vector<weight>(Kernels);
        beta  = etl::dyn_vector<weight>(Kernels);

        mean = etl::dyn_vector<weight>(Kernels);
        var  = etl::dyn_vector<weight>(Kernels);

        last_mean = etl::dyn_vector<weight>(Kernels);
        last_var  = etl::dyn_vector<weight>(Kernels);
        inv_var   = etl::dyn_vector<weight>(Kernels);

        // Initializate the weights
        gamma = 1.0;
        beta = 0.0;
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string([[maybe_unused]] std::string pre = "") {
        return "batch_norm";
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_full_string([[maybe_unused]] std::string pre = "") {
        return "batch_norm";
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    size_t parameters() const noexcept {
        return 4 * Kernels;
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    size_t input_size() const noexcept {
        return Kernels * W * H;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    size_t output_size() const noexcept {
        return Kernels * W * H;
    }

    using base_type::test_forward_batch;
    using base_type::train_forward_batch;

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void forward_batch(Output& output, const Input& input) const {
        test_forward_batch(output, input);
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void test_forward_batch(Output& output, const Input& input) const {
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
    void train_forward_batch(Output& output, const Input& input) {
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
    template <typename C>
    void adapt_errors([[maybe_unused]] C& context) const {}

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
        std::get<0>(context.up.context)->grad = etl::bias_batch_sum_4d(input_pre >> context.errors);

        // Gradients of beta
        std::get<1>(context.up.context)->grad = etl::bias_batch_sum_4d(context.errors);
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename InputType>
    output_one_t prepare_one_output() const {
        return output_one_t(Kernels, W, H);
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename InputType>
    output_t prepare_output(size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(size_t i = 0; i < samples; ++i){
            output.emplace_back(Kernels, W, H);
        }
        return output;
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that needs to be initialized
     */
    template<typename DLayer>
    static void dyn_init(DLayer& /*dyn*/){
        // Nothing to do
    }

    /*!
     * \brief Returns the trainable variables of this layer.
     * \return a tuple containing references to the variables of this layer
     */
    decltype(auto) trainable_parameters(){
        return std::make_tuple(std::ref(gamma), std::ref(beta));
    }

    /*!
     * \brief Returns the trainable variables of this layer.
     * \return a tuple containing references to the variables of this layer
     */
    decltype(auto) trainable_parameters() const {
        return std::make_tuple(std::cref(gamma), std::cref(beta));
    }

    /*!
     * \brief Backup the weights in the secondary weights matrix
     */
    void backup_weights() {
        unique_safe_get(bak_gamma) = gamma;
        unique_safe_get(bak_beta)  = beta;
    }

    /*!
     * \brief Restore the weights from the secondary weights matrix
     */
    void restore_weights() {
        gamma = *bak_gamma;
        beta  = *bak_beta;
    }
};

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<dyn_batch_normalization_4d_layer_impl<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = true;  ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dyn_batch_normalization_4d_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_batch_normalization_4d_layer_impl<Desc>, L> {
    using layer_t          = dyn_batch_normalization_4d_layer_impl<Desc>;            ///< The current layer type
    using weight           = typename layer_t::weight;                          ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::dyn_matrix<weight, 4> input;  ///< A batch of input
    etl::dyn_matrix<weight, 4> output; ///< A batch of output
    etl::dyn_matrix<weight, 4> errors; ///< A batch of errors

    sgd_context(const layer_t& layer)
            : input(batch_size, layer.Kernels, layer.W, layer.H), output(batch_size, layer.Kernels, layer.W, layer.H), errors(batch_size, layer.Kernels, layer.W, layer.H) {}
};

} //end of dll namespace
