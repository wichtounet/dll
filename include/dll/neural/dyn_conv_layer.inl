//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/neural_layer.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Standard dynamic convolutional layer of neural network.
 */
template <typename Desc>
struct dyn_conv_layer final : neural_layer<dyn_conv_layer<Desc>, Desc> {
    using desc      = Desc;                  ///< The descriptor type
    using weight    = typename desc::weight; ///< The weight type
    using this_type = dyn_conv_layer<desc>;  ///< This type
    using base_type = neural_layer<this_type, desc>;

    static constexpr auto activation_function = desc::activation_function;
    static constexpr auto w_initializer       = desc::w_initializer;
    static constexpr auto b_initializer       = desc::b_initializer;

    using input_one_t  = etl::dyn_matrix<weight, 3>; ///< The type for one input
    using output_one_t = etl::dyn_matrix<weight, 3>; ///< The type for one output
    using input_t      = std::vector<input_one_t>;   ///< The type for many input
    using output_t     = std::vector<output_one_t>;  ///< The type for many output

    using w_type = etl::dyn_matrix<weight, 4>;
    using b_type = etl::dyn_matrix<weight, 1>;

    //Weights and biases
    w_type w; //!< Weights
    b_type b; //!< Hidden biases

    //Backup weights and biases
    std::unique_ptr<w_type> bak_w; //!< Backup Weights
    std::unique_ptr<b_type> bak_b; //!< Backup Hidden biases

    size_t nv1; ///< The first visible dimension
    size_t nv2; ///< The second visible dimension
    size_t nh1; ///< The first output dimension
    size_t nh2; ///< The second output dimension
    size_t nc;  ///< The number of input channels
    size_t k;   ///< The number of filters

    size_t nw1; ///< The first dimension of the filters
    size_t nw2; ///< The second dimension of the filters

    dyn_conv_layer(): base_type() {
        // Nothing else to init
    }

    void init_layer(size_t nc, size_t nv1, size_t nv2, size_t k, size_t nw1, size_t nw2){
        this->nv1 = nv1;
        this->nv2 = nv2;
        this->nw1 = nw1;
        this->nw2 = nw2;
        this->nc = nc;
        this->k = k;

        this->nh1 = nv1 - nw1 + 1;
        this->nh2 = nv2 - nw2 + 1;

        w = etl::dyn_matrix<weight, 4>(k, nc, nw1, nw2);

        b = etl::dyn_vector<weight>(k);

        initializer_function<w_initializer>::initialize(w, input_size(), output_size());
        initializer_function<b_initializer>::initialize(b, input_size(), output_size());
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    size_t input_size() const noexcept {
        return nc * nv1 * nv2;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    size_t output_size() const noexcept {
        return k * nh1 * nh2;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    size_t parameters() const noexcept {
        return k * nw1 * nw2;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_short_string() const {
        char buffer[1024];
        snprintf(buffer, 1024, "Conv(dyn): %lux%lux%lu -> (%lux%lux%lu) -> %s -> %lux%lux%lu", nc, nv1, nv2, k, nw1, nw2, to_string(activation_function).c_str(), k, nh1, nh2);
        return {buffer};
    }

    using base_type::activate_hidden;

    template <typename V>
    void activate_hidden(output_one_t& output, const V& v) const {
        auto b_rep = etl::force_temporary(etl::rep(b, nh1, nh2));

        etl::reshape(output, 1, k, nh1, nh2) = etl::ml::convolution_forward(etl::reshape(v, 1, nc, nv1, nv2), w);

        output = f_activate<activation_function>(b_rep + output);
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \return A batch of output corresponding to the activated input
     */
    template <typename V>
    auto batch_activate_hidden(const V& v) const {
        const auto Batch = etl::dim<0>(v);
        etl::dyn_matrix<weight, 4> output(Batch, k, nh1, nh2);
        batch_activate_hidden(output, v);
        return output;
    }

    template <typename H1, typename V, cpp_enable_if(etl::dimensions<V>() == 4)>
    void batch_activate_hidden(H1&& output, const V& v) const {
        dll::auto_timer timer("conv:forward_batch");

        output = etl::ml::convolution_forward(v, w);
        output = f_activate<activation_function>(bias_add_4d(output, b));
    }

    template <typename H1, typename V, cpp_enable_if(etl::dimensions<V>() == 2)>
    void batch_activate_hidden(H1&& output, const V& v) const {
        dll::auto_timer timer("conv:forward_batch");

        output = etl::ml::convolution_forward(etl::reshape(v, etl::dim<0>(v), nc, nv1, nv2), w);
        output = f_activate<activation_function>(bias_add_4d(output, b));
    }

    void prepare_input(input_one_t& input) const {
        input = input_one_t(nc, nv1, nv2);
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    output_t prepare_output(size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(size_t i = 0; i < samples; ++i){
            output.emplace_back(k, nh1, nh2);
        }
        return output;
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    output_one_t prepare_one_output() const {
        return output_one_t(k, nh1, nh2);
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
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
        dll::auto_timer timer("conv:adapt_errors");

        if(activation_function != function::IDENTITY){
            context.errors = f_derivative<activation_function>(context.output) >> context.errors;
        }
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        dll::auto_timer timer("conv:backward_batch");

        output = etl::ml::convolution_backward(context.errors, w);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        dll::auto_timer timer("conv:compute_gradients");

        context.w_grad = etl::ml::convolution_backward_filter(context.input, context.errors);
        context.b_grad = etl::bias_batch_sum_4d(context.errors);
    }
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<dyn_conv_layer<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false;  ///< Indicates if the layer is dense
    static constexpr bool is_conv       = true; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false;  ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = false; ///< Indicates if the layer is a patches layer
    static constexpr bool is_augment    = false; ///< Indicates if the layer is an augment layer
    static constexpr bool is_dynamic    = true; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dync_conv_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_conv_layer<Desc>, L> {
    using layer_t = dyn_conv_layer<Desc>;
    using weight  = typename layer_t::weight;

    static constexpr auto batch_size = DBN::batch_size;

    etl::dyn_matrix<weight, 4> w_grad;
    etl::dyn_matrix<weight, 1> b_grad;

    etl::dyn_matrix<weight, 4> w_inc;
    etl::dyn_matrix<weight, 1> b_inc;

    etl::dyn_matrix<weight, 4> input;
    etl::dyn_matrix<weight, 4> output;
    etl::dyn_matrix<weight, 4> errors;

    sgd_context(layer_t& layer)
            : w_grad(layer.k, layer.nc, layer.nv1 - layer.nh1 + 1, layer.nv2 - layer.nh2 + 1), b_grad(layer.k),
              w_inc(layer.k, layer.nc, layer.nv1 - layer.nh1 + 1, layer.nv2 - layer.nh2 + 1), b_inc(layer.k),
              input(batch_size, layer.nc, layer.nv1, layer.nv2),
              output(batch_size, layer.k, layer.nh1, layer.nh2), errors(batch_size, layer.k, layer.nh1, layer.nh2) {}
};

} //end of dll namespace
