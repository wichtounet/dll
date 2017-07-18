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
struct batch_normalization_2d_layer : neural_layer<batch_normalization_2d_layer<Desc>, Desc> {
    using desc      = Desc;                                                   ///< The descriptor type
    using base_type = neural_layer<batch_normalization_2d_layer<Desc>, Desc>; ///< The base type
    using weight    = typename desc::weight;                                  ///< The data type of the layer

    static constexpr size_t Input = desc::Input; ///< The input size
    static constexpr weight e     = 1e-8;        ///< Epsilon for numerical stability

    etl::fast_matrix<weight, Input> gamma;
    etl::fast_matrix<weight, Input> beta;

    etl::fast_matrix<weight, Input> mean;
    etl::fast_matrix<weight, Input> var;

    etl::fast_matrix<weight, Input> last_mean;
    etl::fast_matrix<weight, Input> last_var;
    etl::fast_matrix<weight, Input> inv_var;

    etl::dyn_matrix<weight, 2> input_pre; /// B x Input

    weight momentum = 0.9;

    // For SGD
    etl::fast_matrix<weight, Input>& w = gamma;
    etl::fast_matrix<weight, Input>& b = beta;

    //Backup gamma and beta
    std::unique_ptr<etl::fast_matrix<weight, Input>> bak_gamma; ///< Backup gamma
    std::unique_ptr<etl::fast_matrix<weight, Input>> bak_beta;  ///< Backup beta

    batch_normalization_2d_layer() : base_type() {
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
        return 4 * Input;
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    static constexpr size_t input_size() noexcept {
        return Input;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    static constexpr size_t output_size() noexcept {
        return Input;
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

    /*!
     * \brief Initialize the dynamic version of the layer from the fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that needs to be initialized
     */
    template<typename DLayer>
    static void dyn_init(DLayer& dyn){
        dyn.init_layer(Input);
    }
};

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<batch_normalization_2d_layer<Desc>> {
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
 * \brief Specialization of sgd_context for batch_normalization_2d_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, batch_normalization_2d_layer<Desc>, L> {
    using layer_t = batch_normalization_2d_layer<Desc>; ///< The current layer type
    using weight  = typename layer_t::weight;           ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, Desc::Input> input;  ///< A batch of input
    etl::fast_matrix<weight, batch_size, Desc::Input> output; ///< A batch of output
    etl::fast_matrix<weight, batch_size, Desc::Input> errors; ///< A batch of errors

    etl::fast_matrix<weight, Desc::Input> w_grad;
    etl::fast_matrix<weight, Desc::Input> b_grad;

    sgd_context(layer_t& /*layer*/){}
};

} //end of dll namespace
