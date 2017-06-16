//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/transform/transform_layer.hpp"

namespace dll {

/*!
 * \brief Batch Normalization layer
 */
template <typename Desc>
struct batch_normalization_2d_layer : transform_layer<batch_normalization_2d_layer<Desc>> {
    using desc      = Desc;                                             ///< The descriptor type
    using base_type = transform_layer<batch_normalization_2d_layer<Desc>>; ///< The base type

    static constexpr size_t Input = desc::Input; ///< The input size
    static constexpr float e      = 1e-8;        ///< Epsilon for numerical stability

    batch_normalization_2d_layer() = default;

    etl::fast_matrix<float, Input> gamma;
    etl::fast_matrix<float, Input> beta;

    etl::fast_matrix<float, Input> mean;
    etl::fast_matrix<float, Input> var;

    etl::fast_matrix<float, Input> last_mean;
    etl::fast_matrix<float, Input> last_var;

    etl::fast_matrix<float, Input> input_pre;

    float momentum = 0.9;

    batch_normalization_2d_layer() : transform_layer() {
        gamma = 1.0;
        beta = 1.0;
    }

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "batch_norm";
    }

    using base_type::activate_hidden;

    /*!
     * \brief Apply the layer to the input
     * \param output The output
     * \param input The input to apply the layer to
     */
    template <typename Input, typename Output>
    static void activate_hidden(Output& output, const Input& input) {
        test_activate_hidden(output, input);
    }

    /*!
     * \brief Apply the layer to the input
     * \param output The output
     * \param input The input to apply the layer to
     */
    template <typename Input, typename Output>
    static void test_activate_hidden(Output& output, const Input& input) {
        output = input;
    }

    /*!
     * \brief Apply the layer to the input
     * \param output The output
     * \param input The input to apply the layer to
     */
    template <typename Input, typename Output>
    static void train_activate_hidden(Output& output, const Input& input) {
        output = input;

    }

    /*!
     * \brief Apply the layer to the batch of input
     * \return A batch of output corresponding to the activated input
     */
    template <typename V>
    auto batch_activate_hidden(const V& v) const {
        auto output = force_temporary_dim_only(v);
        batch_activate_hidden(output, v);
        return output;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input) {
        test_batch_activate_hidden(output, input);
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void test_batch_activate_hidden(Output& output, const Input& input) {
        const auto B = etl::dim<0>(input);

        auto mean_rep = etl::rep(mean, B);
        auto var_rep  = etl::rep(var, B);

        output = gamma * (input - mean_rep) / etl::sqrt(var_rep + e) + beta;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void train_batch_activate_hidden(Output& output, const Input& input) {
        const auto B = etl::dim<0>(input);

        last_mean     = etl::force_temporary(etl::mean_l(input));
        auto last_mean_rep = etl::rep(last_mean, B);

        last_var      = etl::force_temporary(etl::mean_l((input - last_mean_rep) >> (input - last_mean_rep)));
        auto last_var_rep  = etl::rep(last_var, B);

        input_pre = (input - last_mean_rep) / etl::sqrt(last_var_rep + e);

        output = gamma * input_pre + beta;

        // Update the current mean and variance
        mean = momentum * mean + (1.0 - momentum) * last_mean;
        var  = momentum * var + (1.0 - momentum) * (B / (B - 1) * last_var);
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
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        const auto B = etl::dim<0>(context.input);

        output = (1.0 / B) * gamma * etl::sqrt(last_var + eps) * (B * context.errors - etl::sum_l(context.errors) - (context.input - last_mean) * (1.0 / (last_var + eps)) * etl::sum_l(context.errors * (context.input - last_mean)));
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        auto dbeta = etl::sum_l(context.errors);
        auto dgamma = etl::sum_l(input_pre >> context.errors);

        cpp_unused(context);
    }
};

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<batch_normalization_2d_layer<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = true;  ///< Indicates if the layer is a transform layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for batch_normalization_2d_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, batch_normalization_2d_layer<Desc>, L> {
    using layer_t          = batch_normalization_2d_layer<Desc>;                            ///< The current layer type
    using previous_layer   = typename DBN::template layer_type<L - 1>;          ///< The previous layer type
    using previous_context = sgd_context<DBN, previous_layer, L - 1>;           ///< The previous layer's context
    using inputs_t         = decltype(std::declval<previous_context>().output); ///< The type of inputs

    inputs_t input;  ///< A batch of input
    inputs_t output; ///< A batch of output
    inputs_t errors; ///< A batch of errors

    sgd_context(layer_t& /*layer*/){}
};

} //end of dll namespace
