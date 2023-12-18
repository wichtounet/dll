//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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
struct batch_normalization_2d_layer_impl : neural_layer<batch_normalization_2d_layer_impl<Desc>, Desc> {
    using desc        = Desc;                                                        ///< The descriptor type
    using base_type   = neural_layer<batch_normalization_2d_layer_impl<Desc>, Desc>; ///< The base type
    using weight      = typename desc::weight;                                       ///< The data type of the layer
    using this_type   = batch_normalization_2d_layer_impl<Desc>;                     ///< The type of this layer
    using layer_t     = this_type;                                                   ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;                                  ///< The dynamic version of this layer

    static constexpr size_t Input = desc::Input; ///< The input size
    static constexpr weight e     = 1e-8;        ///< Epsilon for numerical stability

    using input_one_t  = etl::fast_dyn_matrix<weight, Input>; ///< The type of one input
    using output_one_t = etl::fast_dyn_matrix<weight, Input>; ///< The type of one output
    using input_t      = std::vector<input_one_t>;            ///< The type of the input
    using output_t     = std::vector<output_one_t>;           ///< The type of the output

    etl::fast_matrix<weight, Input> gamma;
    etl::fast_matrix<weight, Input> beta;

    etl::fast_matrix<weight, Input> mean;
    etl::fast_matrix<weight, Input> var;

    etl::fast_matrix<weight, Input> last_mean;
    etl::fast_matrix<weight, Input> last_var;
    etl::fast_matrix<weight, Input> inv_var;

    etl::dyn_matrix<weight, 2> input_pre; /// B x Input

    weight momentum = 0.9;

    //Backup gamma and beta
    std::unique_ptr<etl::fast_matrix<weight, Input>> bak_gamma; ///< Backup gamma
    std::unique_ptr<etl::fast_matrix<weight, Input>> bak_beta;  ///< Backup beta

    batch_normalization_2d_layer_impl() : base_type() {
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
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {Input};
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
        dll::auto_timer timer("bn:2d:test:forward");

        output = batch_hint((1.0 / etl::sqrt(var + e)) >> (input - mean));
        output = batch_hint((gamma >> output) + beta);
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void train_forward_batch(Output& output, const Input& input) {
        dll::auto_timer timer("bn:2d:train:forward");

        const auto B = etl::dim<0>(input);

        last_mean = etl::bias_batch_mean_2d(input);
        last_var  = etl::bias_batch_var_2d(input, last_mean);
        inv_var   = 1.0 / etl::sqrt(last_var + e);

        input_pre.inherit_if_null(input);

        input_pre = batch_hint(inv_var >> (input - last_mean));
        output    = batch_hint((gamma >> input_pre) + beta);

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
    template <typename C>
    void adapt_errors([[maybe_unused]] C& context) const {}

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        dll::unsafe_auto_timer timer("bn:2d:backward");

        const auto B = etl::dim<0>(context.input);

        auto& dgamma = std::get<0>(context.up.context)->grad;
        auto& dbeta  = std::get<1>(context.up.context)->grad;

        dbeta  = bias_batch_sum_2d(context.errors);
        dgamma = bias_batch_sum_2d(input_pre >> context.errors);

        output = batch_hint((dgamma >> input_pre) + dbeta);
        output = batch_hint(((1.0 / B) >> inv_var >> gamma) >> ((B >> context.errors) - output));
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        // If the layer is not the first one, the gradients already have been computed

        if (!C::layer) {
            dll::unsafe_auto_timer timer("bn:2d:gradients");

            // Gradients of gamma
            std::get<0>(context.up.context)->grad = bias_batch_sum_2d(input_pre >> context.errors);

            // Gradients of beta
            std::get<1>(context.up.context)->grad = bias_batch_sum_2d(context.errors);
        }
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename InputType>
    output_one_t prepare_one_output() const {
        return {};
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename InputType>
    static output_t prepare_output(size_t samples) {
        return output_t{samples};
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that needs to be initialized
     */
    template<typename DLayer>
    static void dyn_init(DLayer& dyn){
        dyn.init_layer(Input);
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
struct layer_base_traits<batch_normalization_2d_layer_impl<Desc>> {
    static constexpr bool is_neural     = true; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false;  ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for batch_normalization_2d_layer_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, batch_normalization_2d_layer_impl<Desc>, L> {
    using layer_t = batch_normalization_2d_layer_impl<Desc>; ///< The current layer type
    using weight  = typename layer_t::weight;           ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size; ///< The batch size of the network
    static constexpr auto layer      = L;               ///> The layer's index

    etl::fast_matrix<weight, batch_size, Desc::Input> input;  ///< A batch of input
    etl::fast_matrix<weight, batch_size, Desc::Input> output; ///< A batch of output
    etl::fast_matrix<weight, batch_size, Desc::Input> errors; ///< A batch of errors

    sgd_context(const layer_t& /*layer*/){}
};

} //end of dll namespace
