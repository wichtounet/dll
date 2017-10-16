//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/neural_layer.hpp"

#include "dll/util/timers.hpp" // for auto_timer

namespace dll {

/*!
 * \brief Value traits to compute the addition of all the given values
 */
template <size_t F, size_t... Dims>
struct add_all_impl final : std::integral_constant<size_t, F + add_all_impl<Dims...>::value> {};

/*!
 * \copydoc add_all_impl
 */
template <size_t F>
struct add_all_impl<F> final : std::integral_constant<size_t, F> {};

//CPP17: Can be totally replaced with variadic fold expansion

/*!
 * \brief Value traits to compute the addition of all the given values
 */
template <size_t F, size_t... Dims>
constexpr size_t add_all = add_all_impl<F, Dims...>::value;

template <size_t I, size_t D, typename... Outputs>
struct get_sub_dim {
    static constexpr size_t value = etl::decay_traits<cpp::first_type_t<Outputs...>>::template dim<I>();
};

template <size_t D, typename... Outputs>
struct get_sub_dim<D, D, Outputs...> {
    static constexpr size_t value = add_all<etl::decay_traits<Outputs>::template dim<D>()...>;
};

template <size_t D, typename... Outputs>
struct merge_output_types {
    template <typename T>
    struct build_merged_output;

    template <size_t... I>
    struct build_merged_output<std::index_sequence<I...>> {
        using type = etl::fast_dyn_matrix<etl::value_t<cpp::first_type_t<Outputs...>>, get_sub_dim<I, D, Outputs...>::value...>;
    };

    using type = typename build_merged_output<std::make_index_sequence<etl::decay_traits<cpp::first_type_t<Outputs...>>::dimensions()>>::type;
};

/*!
 * \brief Standard convolutional layer of neural network.
 */
template <size_t D, typename... Layers>
struct merge_layer_impl final : layer<merge_layer_impl<D, Layers...>> {
    static constexpr size_t merge_dim = D; ///< The dimensions at which to merge

    using first_layer_t = cpp::first_type_t<Layers...>; ///< The type of the first layer
    using last_layer_t  = cpp::last_type_t<Layers...>;  ///< The type of the last layer

    using this_type   = merge_layer_impl<D, Layers...>;                       ///< The type of this layer
    using weight      = typename first_layer_t::weight;                       ///< The data type of the layer
    using base_type   = layer<this_type>;                                     ///< The base type of the layer
    using layer_t     = this_type;                                            ///< The type of this layer
    using dyn_layer_t = merge_layer_impl<D, typename Layers::dyn_layer_t...>; ///< The type of this layer

    using input_one_t  = typename first_layer_t::input_one_t;                                    ///< The type of one input
    using output_one_t = typename merge_output_types<D, typename Layers::output_one_t...>::type; ///< The type of one output
    using input_t      = std::vector<input_one_t>;                                               ///< The type of the input
    using output_t     = std::vector<output_one_t>;                                              ///< The type of the output

    static constexpr size_t n_layers = sizeof...(Layers); ///< The number of layers

    std::tuple<Layers...> layers;

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    static constexpr size_t input_size() noexcept {
        return first_layer_t::input_size();
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    static constexpr size_t output_size() noexcept {
        return add_all<Layers::output_size()...>;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    size_t parameters() const noexcept {
        size_t params = 0;

        cpp::for_each(layers, [&params](auto& layer){
            params += layer.parameters();
        });

        return params;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_short_string(std::string pre = "") const {
        std::string str = "Merge(";

        cpp::for_each(layers, [&str, &pre](auto& layer){
            str += "\n" + pre + "  " + layer.to_short_string(pre + "  ");
        });

        str += "\n" + pre + ")";

        return str;
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
        size_t n = 0;
        cpp::for_each(layers, [&n, &input, &output](auto& layer){
            auto sub_output = layer.test_forward_batch(input);

            for(size_t b = 0; b < etl::dim<0>(sub_output); ++b){
                etl::memory_slice(output(b), n, n + etl::size(sub_output(b))) = sub_output(b);
            }

            n += etl::size(sub_output(0));
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
        size_t n = 0;
        cpp::for_each(layers, [&n, &input, &output](auto& layer){
            auto sub_output = layer.train_forward_batch(input);

            for(size_t b = 0; b < etl::dim<0>(sub_output); ++b){
                etl::memory_slice(output(b), n, n + etl::size(sub_output(b))) = sub_output(b);
            }

            n += etl::size(sub_output(0));
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
        size_t n = 0;
        cpp::for_each(layers, [&n, &input, &output](auto& layer){
            auto sub_output = layer.forward_batch(input);

            for(size_t b = 0; b < etl::dim<0>(sub_output); ++b){
                etl::memory_slice(output(b), n, n + etl::size(sub_output(b))) = sub_output(b);
            }

            n += etl::size(sub_output(0));
        });
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     */
    template <typename Input>
    output_one_t prepare_one_output() const {
        return {};
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     */
    template <typename Input>
    static output_t prepare_output(size_t samples) {
        return output_t{samples};
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DynLayer>
    void dyn_init(DynLayer& dyn){
        cpp::for_each_pair(layers, dyn.layers, [](auto& fast_sub, auto& dyn_sub){
            fast_sub.dyn_init(dyn_sub);
        });
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
};

// Declare the traits for the Layer

template<size_t D, typename... Layers>
struct layer_base_traits<merge_layer_impl<D, Layers...>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of the sgd_context for conv_layer_impl
 */
template <typename DBN, size_t D, typename... Layers, size_t L>
struct sgd_context<DBN, merge_layer_impl<D, Layers...>, L> {
    using layer_t = merge_layer_impl<D, Layers...>;

    using input_type  = decltype(std::declval<sgd_context<DBN, cpp::first_type_t<Layers...>, L>>().input);
    using output_type = typename merge_output_types<D + 1, decltype(std::declval<sgd_context<DBN, Layers, L>>().output)...>::type;

    input_type input;
    output_type output;
    output_type errors;

    sgd_context(const merge_layer_impl<D, Layers...>& /* layer */)
            : output(0.0), errors(0.0) {}
};

} //end of dll namespace
