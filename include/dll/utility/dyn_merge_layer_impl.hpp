//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/neural_layer.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Standard merge layer of neural network.
 */
template <size_t D, typename... Layers>
struct dyn_merge_layer_impl <dyn_merge_layer_desc<D, Layers...>> final : layer<dyn_merge_layer_impl<dyn_merge_layer_desc<D, Layers...>>> {
    static constexpr size_t merge_dim = D; ///< The dimensions at which to merge

    using first_layer_t = cpp::first_type_t<Layers...>; ///< The type of the first layer
    using last_layer_t  = cpp::last_type_t<Layers...>;  ///< The type of the last layer

    using desc        = dyn_merge_layer_desc<D, Layers...>; ///< The layer descriptor
    using this_type   = dyn_merge_layer_impl<desc>;         ///< The type of this layer
    using weight      = typename first_layer_t::weight; ///< The data type of the layer
    using base_type   = layer<this_type>;               ///< The base type of the layer
    using layer_t     = this_type;                      ///< The type of this layer
    using dyn_layer_t = typename desc::dyn_layer_t;     ///< The type of this layer

    using input_one_t  = typename first_layer_t::input_one_t; ///< The type of one input
    using output_one_t = typename first_layer_t::output_one_t; ///< The type of one output
    using input_t      = std::vector<input_one_t>;            ///< The type of the input
    using output_t     = std::vector<output_one_t>;           ///< The type of the output

    static constexpr size_t n_layers = sizeof...(Layers); ///< The number of layers

    std::tuple<Layers...> layers; ///< The layers to merge

    /*!
     * \brief Return the type of the Lth layer
     * \tparam L The layer index
     */
    template<size_t L>
    using layer_type = cpp::nth_type_t<L, Layers...>;

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    size_t input_size() const noexcept {
        return std::get<0>(layers).input_size();
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    size_t output_size() const noexcept {
        size_t out_size = 0;

        cpp::for_each(layers, [&out_size](auto& layer){
            out_size += layer.output_size();
        });

        return out_size;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    size_t parameters() const noexcept {
        size_t param = 0;

        cpp::for_each(layers, [&param](auto& layer){
            param += layer.parameters();
        });

        return param;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_full_string(std::string pre = "") const {
        std::string str = "Merge(";

        cpp::for_each(layers, [&str, &pre](auto& layer){
            str += "\n" + pre + "  " + layer.to_full_string(pre + "  ");
        });

        str += "\n" + pre + ")";

        return str;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_short_string([[maybe_unused]] std::string pre = "") const {
        return "Merge";
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape(const std::vector<size_t>& input_shape) const {
        std::vector<size_t> output;

        cpp::for_each(layers, [&](auto& layer) {
            if (output.empty()) {
                output = layer.output_shape(input_shape);
            } else {
                auto next = layer.output_shape(input_shape);
                output[D] += next[D];
            }
        });

        return output;
    }

    using base_type::forward_batch;
    using base_type::train_forward_batch;
    using base_type::test_forward_batch;

    /*!
     * \brief Apply the layer to the given batch of input.
     *
     * \param input A batch of input
     * \param output A batch of output that will be filled
     */
    template <typename H1, typename V>
    void test_forward_batch(H1&& output, const V& input) const {
        cpp::for_each_i(layers, [&input, &output](size_t i, auto& layer){
            auto sub_output = layer.test_forward_batch(input);

            etl::batch_merge(output, sub_output, i);
        });
    }

    /*!
     * \brief Apply the layer to the given batch of input.
     *
     * \param input A batch of input
     * \param output A batch of output that will be filled
     */
    template <typename H1, typename V>
    void train_forward_batch(H1&& output, const V& input) const {
        cpp::for_each_i(layers, [&input, &output](size_t i, auto& layer){
            auto sub_output = layer.train_forward_batch(input);

            etl::batch_merge(output, sub_output, i);
        });
    }

    /*!
     * \brief Apply the layer to the given batch of input.
     *
     * \param input A batch of input
     * \param output A batch of output that will be filled
     */
    template <typename H1, typename V>
    void forward_batch(H1&& output, const V& input) const {
        cpp::for_each_i(layers, [&input, &output](size_t i, auto& layer){
            auto sub_output = layer.forward_batch(input);

            etl::batch_merge(output, sub_output, i);
        });
    }

    template <typename Input, size_t I>
    size_t get_sub_dim() const {
        if constexpr (I == D) {
            size_t v = 0;

            cpp::for_each(layers, [&v](auto& layer) {
                v += etl::dim<I>(layer.template prepare_one_output<Input>());
            });

            return v;
        } else {
            return etl::dim<I>(std::get<0>(layers).template prepare_one_output<Input>());
        }
    }

    template <typename Input, size_t... I>
    output_one_t prepare_one_output_impl(std::index_sequence<I...> /*seq*/) const {
        return output_one_t(get_sub_dim<Input, I>()...);
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     */
    template <typename Input>
    output_one_t prepare_one_output() const {
        return prepare_one_output_impl<Input>(std::make_index_sequence<etl::dimensions<output_one_t>()>());
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     */
    template <typename Input>
    output_t prepare_output(size_t samples) {
        output_t out{samples};
        for(size_t i = 0; i < samples; ++i){
            out[i] = prepare_one_output<Input>();
        }
        return out;
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DynLayer>
    static void dyn_init([[maybe_unused]] DynLayer& dyn){
        // Nothing to do here
    }

    /*!
     * \brief Backup the weights in the secondary weights matrix
     */
    void backup_weights() {
        cpp::for_each(layers, [](auto& layer) {
            layer.backup_weights();
        });
    }

    /*!
     * \brief Restore the weights from the secondary weights matrix
     */
    void restore_weights() {
        cpp::for_each(layers, [](auto& layer) {
            layer.restore_weights();
        });
    }

    /*!
     * \brief Return the Lth layer
     * \tparam L The layer index
     */
    template<size_t L>
    decltype(auto) layer_get(){
        return std::get<L>(layers);
    }

    /*!
     * \brief Return the Lth layer
     * \tparam L The layer index
     */
    template<size_t L>
    decltype(auto) layer_get() const {
        return std::get<L>(layers);
    }
};

// Declare the traits for the Layer

template<size_t D, typename... Layers>
struct layer_base_traits<dyn_merge_layer_impl<dyn_merge_layer_desc<D, Layers...>>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = true;  ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = true;  ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of the sgd_context for conv_layer_impl
 */
template <typename DBN, size_t D, typename... Layers, size_t L>
struct sgd_context<DBN, dyn_merge_layer_impl<dyn_merge_layer_desc<D, Layers...>>, L> {
    using layer_t = dyn_merge_layer_impl<dyn_merge_layer_desc<D, Layers...>>;

    using input_type  = decltype(std::declval<sgd_context<DBN, cpp::first_type_t<Layers...>, L>>().input);
    using output_type = decltype(std::declval<sgd_context<DBN, cpp::first_type_t<Layers...>, L>>().output);

    input_type input;
    output_type output;
    output_type errors;

    sgd_context(const layer_t& layer) {
        sgd_context<DBN, cpp::first_type_t<Layers...>, L> first_context(std::get<0>(layer.layers));
        input = first_context.input;

        output = batch_extend(input, layer.template prepare_one_output<decltype(input(0))>());
        errors = batch_extend(input, layer.template prepare_one_output<decltype(input(0))>());
    }
};

} //end of dll namespace
