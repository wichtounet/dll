//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/neural_layer.hpp"

namespace dll {

/*!
 * \brief Standard dynamic convolutional layer of neural network.
 */
template <typename Desc>
struct dyn_conv_same_layer final : neural_layer<dyn_conv_same_layer<Desc>, Desc> {
    using desc      = Desc;                          ///< The descriptor type
    using weight    = typename desc::weight;         ///< The weight type
    using this_type = dyn_conv_same_layer<desc>;     ///< This type
    using base_type = neural_layer<this_type, desc>; ///< The base type

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

    size_t p1; ///< The first dimension padding
    size_t p2; ///< The second dimension padding

    dyn_conv_same_layer(): base_type() {
        // Nothing else to init
    }

    void init_layer(size_t nc, size_t nv1, size_t nv2, size_t k, size_t nw1, size_t nw2){
        this->nv1 = nv1;
        this->nv2 = nv2;
        this->nw1 = nw1;
        this->nw2 = nw2;
        this->nc = nc;
        this->k = k;

        this->nh1 = nv1;
        this->nh2 = nv2;

        this->p1 = (nw1 - 1) / 2;
        this->p2 = (nw2 - 1) / 2;

        w = etl::dyn_matrix<weight, 4>(k, nc, nw1, nw2);

        b = etl::dyn_vector<weight>(k);

        initializer_function<w_initializer>::initialize(w, input_size(), output_size());
        initializer_function<b_initializer>::initialize(b, input_size(), output_size());
    }

    std::size_t input_size() const noexcept {
        return nc * nv1 * nv2;
    }

    std::size_t output_size() const noexcept {
        return k * nh1 * nh2;
    }

    std::size_t parameters() const noexcept {
        return k * nw1 * nw2;
    }

    std::string to_short_string() const {
        char buffer[1024];
        snprintf(buffer, 1024, "Conv(Same,dyn): %lux%lux%lu -> (%lux%lux%lu) -> %s -> %lux%lux%lu", nc, nv1, nv2, k, nw1, nw2, to_string(activation_function).c_str(), k, nh1, nh2);
        return {buffer};
    }

    using base_type::activate_hidden;

    template <typename V>
    void activate_hidden(output_one_t& output, const V& v) const {
        auto b_rep = etl::force_temporary(etl::rep(b, nh1, nh2));

        etl::reshape(output, 1, k, nh1, nh2) = etl::conv_4d_valid_flipped(etl::reshape(v, 1, nc, nv1, nv2), w, 1, 1, p1, p2);

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

    template <typename H1, typename V>
    void batch_activate_hidden(H1&& output, const V& v) const {
        output = etl::conv_4d_valid_flipped(v, w, 1, 1, p1, p2);

        auto b_rep = etl::force_temporary(etl::rep_l(etl::rep(b, nh1, nh2), etl::dim<0>(output)));

        output = f_activate<activation_function>(b_rep + output);
    }

    void prepare_input(input_one_t& input) const {
        input = input_one_t(nc, nv1, nv2);
    }

    template <typename Input>
    output_t prepare_output(std::size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(size_t i = 0; i < samples; ++i){
            output.emplace_back(k, nh1, nh2);
        }
        return output;
    }

    template <typename Input>
    output_one_t prepare_one_output() const {
        return output_one_t(k, nh1, nh2);
    }

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
        context.errors = f_derivative<activation_function>(context.output) >> context.errors;
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        output = conv_4d_valid_back_flipped(context.errors, w, 1, 1, p1, p2);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        context.w_grad = conv_4d_valid_filter_flipped(context.input, context.errors, 1, 1, p1, p2);
        context.b_grad = bias_batch_mean(context.errors);
    }
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<dyn_conv_same_layer<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = true;  ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = false; ///< Indicates if the layer is a patches layer
    static constexpr bool is_augment    = false; ///< Indicates if the layer is an augment layer
    static constexpr bool is_dynamic    = true;  ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for dync_conv_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_conv_same_layer<Desc>, L> {
    using layer_t = dyn_conv_same_layer<Desc>;
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
            : w_grad(layer.k, layer.nc, layer.nw1, layer.nw2), b_grad(layer.k),
              w_inc(layer.k, layer.nc, layer.nv1, layer.nv2), b_inc(layer.k),
              input(batch_size, layer.nc, layer.nv1, layer.nv2),
              output(batch_size, layer.k, layer.nv1, layer.nv2), errors(batch_size, layer.k, layer.nv1, layer.nv2) {}
};

} //end of dll namespace
