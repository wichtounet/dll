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

    etl::fast_matrix<float, Input> gamma;
    etl::fast_matrix<float, Input> beta;

    etl::fast_matrix<float, Input> mean;
    etl::fast_matrix<float, Input> var;

    etl::fast_matrix<float, Input> last_mean;
    etl::fast_matrix<float, Input> last_var;
    etl::fast_matrix<float, Input> inv_var;

    etl::dyn_matrix<float, 2> input_pre; /// B x Input

    float momentum = 0.9;

    etl::fast_matrix<float, Input>& w = gamma;
    etl::fast_matrix<float, Input>& b = beta;

    batch_normalization_2d_layer() : base_type() {
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

        // TODO
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \return A batch of output corresponding to the activated input
     */
    template <typename V>
    auto batch_activate_hidden(const V& v) const {
        auto output = force_temporary_dim_only(v);
        test_batch_activate_hidden(output, v);
        return output;
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \return A batch of output corresponding to the activated input
     */
    template <typename V>
    auto test_batch_activate_hidden(const V& v) const {
        auto output = force_temporary_dim_only(v);
        test_batch_activate_hidden(output, v);
        return output;
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

        for(size_t b = 0; b < B; ++b){
            output(b) = (gamma >> ((input(b) - mean) >> inv_var)) + beta;
        }
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void train_batch_activate_hidden(Output& output, const Input& input) {
        const auto B = etl::dim<0>(input);

        last_mean          = etl::mean_l(input);
        auto last_mean_rep = etl::rep_l(last_mean, B);

        last_var = etl::mean_l((input - last_mean_rep) >> (input - last_mean_rep));
        inv_var  = 1.0 / etl::sqrt(last_var + e);

        input_pre.inherit_if_null(input);

        for(size_t b = 0; b < B; ++b){
            input_pre(b) = (input(b) - last_mean) >> inv_var;
            output(b)    = (gamma >> input_pre(b)) + beta;
        }

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

        auto dxhat        = etl::force_temporary(context.errors >> etl::rep_l(gamma, B));
        auto dxhat_l      = etl::force_temporary(etl::sum_l(dxhat));
        auto dxhat_xhat_l = etl::force_temporary(etl::sum_l(dxhat >> input_pre));

        for(size_t b = 0; b < B; ++b){
            output(b) = (1.0 / B) >> inv_var >> (B * dxhat(b) - dxhat_l - (input_pre(b) >> dxhat_xhat_l));
        }
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        // Gradients of gamma
        context.w_grad = etl::sum_l(input_pre >> context.errors);

        // Gradients of beta
        context.b_grad = etl::sum_l(context.errors);
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

    etl::fast_matrix<float, Desc::Input> w_grad;
    etl::fast_matrix<float, Desc::Input> b_grad;

    sgd_context(layer_t& /*layer*/){}
};

} //end of dll namespace
