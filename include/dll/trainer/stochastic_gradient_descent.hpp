//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file stochastic_gradient_descent.hpp
 * \brief Stochastic Gradient Descent (SGD) Implementation for neural networks
 *
 * This implementations supports fully-connected layers, convolutional layers,
 * RBM layers, CRBM layers, transform layers and pooling layers.
 */

#pragma once

#include "cpp_utils/tuple_utils.hpp"

#include "dll/trainer/context_fwd.hpp" // For sgd_context
#include "dll/util/checks.hpp"         // For NaN checks
#include "dll/util/timers.hpp"         // For auto_timer

namespace dll {

template <typename Layer>
static constexpr bool is_group_layer = cpp::is_specialization_of_v<dll::group_layer_impl, Layer> || cpp::is_specialization_of_v<dll::dyn_group_layer_impl, Layer>;

template <typename Layer>
static constexpr bool is_merge_layer = cpp::is_specialization_of_v<dll::merge_layer_impl, Layer> || cpp::is_specialization_of_v<dll::dyn_merge_layer_impl, Layer>;

template <typename Layer>
static constexpr bool is_utility_layer = is_group_layer<Layer> || is_merge_layer<Layer>;

/*!
 * \brief Build the sub context for a updater context
 *
 * \param layer The layer to build the context for
 */
template <template <typename, size_t, updater_type> typename SubContext, updater_type UT, typename Layer, size_t... I>
auto build_sub_context(const Layer& layer, std::index_sequence<I...> /*seq*/) {
    return std::make_tuple
        (
            std::make_shared<SubContext<Layer, I, UT>>(layer)...
        );
}

/*!
 * \brief Build the sub context for a updater context
 *
 * \param layer The layer to build the context for
 */
template <template <typename, size_t, updater_type> typename SubContext, updater_type UT, typename Layer>
auto build_sub_context(const Layer& layer) {
    static constexpr size_t N = std::tuple_size<decltype(std::declval<Layer>().trainable_parameters())>();

    return build_sub_context<SubContext, UT>(layer, std::make_index_sequence<N>());
}

/*!
 * \brief The sub context for a specific updater
 * \param Layer The layer to optimize
 * \param I The index of the variable to optimize
 * \param UT The updater type
 */
template<typename Layer, size_t I, updater_type UT>
struct updater_sub_context;

/*!
 * \brief Specialization of updater_sub_context for SGD updater
 */
template<typename Layer, size_t I>
struct updater_sub_context <Layer, I, updater_type::SGD> {
    /*!
     * \brief The type of the variable to optimize
     */
    using type = std::remove_reference_t<decltype(std::get<I>(std::declval<Layer>().trainable_parameters()))>;

    type grad; ///< The gradients of the variable

    /*!
     * \brief Construct the sub_context for the given layer
     * \param layer The layer to build the context for
     */
    updater_sub_context(const Layer& layer) : grad(std::get<I>(layer.trainable_parameters())) {
        grad = 0;
    }
};

/*!
 * \brief Specialization of updater_sub_context for momentum updater
 */
template<typename Layer, size_t I>
struct updater_sub_context <Layer, I, updater_type::MOMENTUM> {
    /*!
     * \brief The type of the variable to optimize
     */
    using type = std::remove_reference_t<decltype(std::get<I>(std::declval<Layer>().trainable_parameters()))>;

    type grad; ///< The gradients of the variable
    type inc;  ///< The accumulated momentum cache

    /*!
     * \brief Construct the sub_context for the given layer
     * \param layer The layer to build the context for
     */
    updater_sub_context(const Layer& layer) : grad(std::get<I>(layer.trainable_parameters())), inc(grad) {
        grad = 0;
        inc = 0;
    }
};

/*!
 * \brief Specialization of updater_sub_context for Nesterov Accelerated Gradients updater
 */
template<typename Layer, size_t I>
struct updater_sub_context <Layer, I, updater_type::NESTEROV> {
    /*!
     * \brief The type of the variable to optimize
     */
    using type = std::remove_reference_t<decltype(std::get<I>(std::declval<Layer>().trainable_parameters()))>;

    type grad;     ///< The gradients of the variable
    type inc;      ///< The accumulated momentum cache
    type inc_prev; ///< The previous accumulated momentum cache

    /*!
     * \brief Construct the sub_context for the given layer
     * \param layer The layer to build the context for
     */
    updater_sub_context(const Layer& layer) : grad(std::get<I>(layer.trainable_parameters())), inc(grad), inc_prev(grad) {
        grad = 0;
        inc = 0;
        inc_prev = 0;
    }
};

/*!
 * \brief Specialization of updater_sub_context for RMSPROP updater
 */
template<typename Layer, size_t I>
struct updater_sub_context <Layer, I, updater_type::RMSPROP> {
    /*!
     * \brief The type of the variable to optimize
     */
    using type = std::remove_reference_t<decltype(std::get<I>(std::declval<Layer>().trainable_parameters()))>;

    type grad; ///< The gradients of the variable
    type inc;  ///< The accumulated squared gradients

    /*!
     * \brief Construct the sub_context for the given layer
     * \param layer The layer to build the context for
     */
    updater_sub_context(const Layer& layer) : grad(std::get<I>(layer.trainable_parameters())), inc(grad) {
        grad = 0;
        inc = 0;
    }
};

/*!
 * \brief Specialization of updater_sub_context for Adagrad updater
 */
template<typename Layer, size_t I>
struct updater_sub_context <Layer, I, updater_type::ADAGRAD> {
    /*!
     * \brief The type of the variable to optimize
     */
    using type = std::remove_reference_t<decltype(std::get<I>(std::declval<Layer>().trainable_parameters()))>;

    type grad; ///< The gradients of the variable
    type inc;  ///< Accumulated gradients for adagrad

    /*!
     * \brief Construct the sub_context for the given layer
     * \param layer The layer to build the context for
     */
    updater_sub_context(const Layer& layer) : grad(std::get<I>(layer.trainable_parameters())), inc(grad) {
        grad = 0;
        inc = 0;
    }
};

/*!
 * \brief Specialization of updater_sub_context for Adadelta updater
 */
template<typename Layer, size_t I>
struct updater_sub_context <Layer, I, updater_type::ADADELTA> {
    /*!
     * \brief The type of the variable to optimize
     */
    using type = std::remove_reference_t<decltype(std::get<I>(std::declval<Layer>().trainable_parameters()))>;

    type grad; ///< The gradients of the variable
    type g;
    type x;
    type v;

    /*!
     * \brief Construct the sub_context for the given layer
     * \param layer The layer to build the context for
     */
    updater_sub_context(const Layer& layer) : grad(std::get<I>(layer.trainable_parameters())), g(grad), x(grad), v(grad) {
        grad = 0;
        g = 0;
        x = 0;
        v = 0;
    }
};

/*!
 * \brief The context for the Adam updater
 */
template<typename Layer, size_t I>
struct updater_sub_context <Layer, I, updater_type::ADAM> {
    /*!
     * \brief The type of the variable to optimize
     */
    using type = std::remove_reference_t<decltype(std::get<I>(std::declval<Layer>().trainable_parameters()))>;

    type grad; ///< The gradients of the variable
    type m;    ///< Estimates of the first moment of the gradient
    type v;    ///< Estimates of the second moment of the gradient

    /*!
     * \brief Construct the sub_context for the given layer
     * \param layer The layer to build the context for
     */
    updater_sub_context(const Layer& layer) : grad(std::get<I>(layer.trainable_parameters())), m(grad), v(grad) {
        grad = 0;
        m = 0;
        v = 0;
    }
};

/*!
 * \brief The context for the Adam updater with bias correction
 */
template<typename Layer, size_t I>
struct updater_sub_context <Layer, I, updater_type::ADAM_CORRECT> {
    /*!
     * \brief The type of the variable to optimize
     */
    using type = std::remove_reference_t<decltype(std::get<I>(std::declval<Layer>().trainable_parameters()))>;

    type grad; ///< The gradients of the variable
    type m;    ///< Estimates of the first moment of the gradient
    type mt;   ///< Corrected estimates of the first moment of the gradient
    type v;    ///< Estimates of the second moment of the gradient
    type vt;   ///< Corrected estimates of the second moment of the gradient

    /*!
     * \brief Construct the sub_context for the given layer
     * \param layer The layer to build the context for
     */
    updater_sub_context(const Layer& layer) : grad(std::get<I>(layer.trainable_parameters())), m(grad), mt(grad), v(grad), vt(grad) {
        grad = 0;
        m = 0;
        mt = 0;
        v = 0;
        vt = 0;
    }
};

/*!
 * \brief The context for the Nesterov Adam (NAdam) updater with bias correction
 */
template<typename Layer, size_t I>
struct updater_sub_context <Layer, I, updater_type::NADAM> {
    /*!
     * \brief The type of the variable to optimize
     */
    using type = std::remove_reference_t<decltype(std::get<I>(std::declval<Layer>().trainable_parameters()))>;

    type grad; ///< The gradients of the variable
    type m;    ///< Estimates of the first moment of the gradient
    type v;    ///< Estimates of the second moment of the gradient

    // Simplified for performance's sake
    // type mt;   ///< Corrected estimates of the first moment of the gradient
    // type vt;   ///< Corrected estimates of the second moment of the gradient

    double m_schedule = 1.0;

    /*!
     * \brief Construct the sub_context for the given layer
     * \param layer The layer to build the context for
     */
    updater_sub_context(const Layer& layer) : grad(std::get<I>(layer.trainable_parameters())), m(grad), v(grad) {
        grad = 0;
        m = 0;
        v = 0;
    }
};

/*!
 * \brief The context for the Adamax updater
 */
template<typename Layer, size_t I>
struct updater_sub_context <Layer, I, updater_type::ADAMAX> {
    /*!
     * \brief The type of the variable to optimize
     */
    using type = std::remove_reference_t<decltype(std::get<I>(std::declval<Layer>().trainable_parameters()))>;

    type grad; ///< The gradients of the variable
    type m;    ///< Estimates of the first moment of the gradient
    type v;    ///< Estimates of the second moment of the gradient

    /*!
     * \brief Construct the sub_context for the given layer
     * \param layer The layer to build the context for
     */
    updater_sub_context(const Layer& layer) : grad(std::get<I>(layer.trainable_parameters())), m(grad), v(grad) {
        grad = 0;
        m = 0;
        v = 0;
    }
};


/*!
 * \brief The context for the base updater (no update).
 */
template <updater_type UT, bool Neural, typename Layer>
struct updater_context {
    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context([[maybe_unused]] const Layer& layer) {}
};

/*!
 * \brief The context for the real updaters.
 */
template <updater_type UT, typename Layer>
struct updater_context<UT, true, Layer> {
    /*!
     * \brief The context for the updater and for each variable of the layer
     */
    decltype(build_sub_context<updater_sub_context, UT>(std::declval<Layer&>())) context;

    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context(const Layer& layer) : context(build_sub_context<updater_sub_context, UT>(layer)) {
        // Nothing else to init
    }
};

/*!
 * \brief The full SGD context, it contains the context of the layer as well as
 * the context for the SGD updater
 */
template <typename DBN, typename Layer, size_t L>
struct full_sgd_context : sgd_context<DBN, Layer, L> {
    using context_type = sgd_context<DBN, Layer, L>; ///< The parent context type

    /*!
     * \brief The updater context
     */
    updater_context<DBN::updater, decay_layer_traits<Layer>::is_neural_layer(), Layer> up;

    /*!
     * \brief Construct the full_sgd_context for the given layer
     */
    full_sgd_context(const Layer& layer) : context_type(layer), up(layer) {
        // Nothing else to init
    }
};

/*!
 * \brief The full SGD context, it contains the context of the layer as well as
 * the context for the SGD updater
 */
template <typename DBN, typename... Layers, size_t L>
struct full_sgd_context <DBN, group_layer_impl<group_layer_desc<Layers...>>, L>  {
    using layer_t      = group_layer_impl<group_layer_desc<Layers...>>; ///< The layer
    using context_type = sgd_context<DBN, layer_t, L>;                  ///< The parent context type

    static constexpr size_t n_layers = sizeof...(Layers); ///< The number of layers

    std::tuple<full_sgd_context<DBN, Layers, L>...> sub_contexts; ///< The sub contexts

    /*!
     * \brief Construct the full_sgd_context for the given layer
     */
    full_sgd_context(const layer_t& layer) : sub_contexts(layer.layers) {
        // Nothing else to init
    }
};

/*!
 * \brief The full SGD context, it contains the context of the layer as well as
 * the context for the SGD updater
 */
template <typename DBN, typename... Layers, size_t L>
struct full_sgd_context <DBN, dyn_group_layer_impl<dyn_group_layer_desc<Layers...>>, L>  {
    using layer_t      = dyn_group_layer_impl<dyn_group_layer_desc<Layers...>>; ///< The layer
    using context_type = sgd_context<DBN, layer_t, L>;                  ///< The parent context type

    static constexpr size_t n_layers = sizeof...(Layers); ///< The number of layers

    std::tuple<full_sgd_context<DBN, Layers, L>...> sub_contexts; ///< The sub contexts

    /*!
     * \brief Construct the full_sgd_context for the given layer
     */
    full_sgd_context(const layer_t& layer) : sub_contexts(layer.layers) {
        // Nothing else to init
    }
};

/*!
 * \brief The full SGD context, it contains the context of the layer as well as
 * the context for the SGD updater
 */
template <typename DBN, size_t D, typename... Layers, size_t L>
struct full_sgd_context<DBN, merge_layer_impl<merge_layer_desc<D, Layers...>>, L> : sgd_context<DBN, merge_layer_impl<merge_layer_desc<D, Layers...>>, L> {
    using layer_t      = merge_layer_impl<merge_layer_desc<D, Layers...>>; ///< The layer
    using context_type = sgd_context<DBN, layer_t, L>;                     ///< The parent context type

    static constexpr size_t n_layers = sizeof...(Layers); ///< The number of layers

    std::tuple<full_sgd_context<DBN, Layers, L>...> sub_contexts; ///< The sub contexts

    /*!
     * \brief Construct the full_sgd_context for the given layer
     */
    full_sgd_context(const layer_t& layer) : context_type(layer), sub_contexts(layer.layers) {
        // Nothing else to init
    }
};

/*!
 * \brief The full SGD context, it contains the context of the layer as well as
 * the context for the SGD updater
 */
template <typename DBN, size_t D, typename... Layers, size_t L>
struct full_sgd_context<DBN, dyn_merge_layer_impl<dyn_merge_layer_desc<D, Layers...>>, L> : sgd_context<DBN, dyn_merge_layer_impl<dyn_merge_layer_desc<D, Layers...>>, L> {
    using layer_t      = dyn_merge_layer_impl<dyn_merge_layer_desc<D, Layers...>>; ///< The layer
    using context_type = sgd_context<DBN, layer_t, L>;                     ///< The parent context type

    static constexpr size_t n_layers = sizeof...(Layers); ///< The number of layers

    std::tuple<full_sgd_context<DBN, Layers, L>...> sub_contexts; ///< The sub contexts

    /*!
     * \brief Construct the full_sgd_context for the given layer
     */
    full_sgd_context(const layer_t& layer) : context_type(layer), sub_contexts(layer.layers) {
        // Nothing else to init
    }
};

/*!
 * \brief Build the context for a DBN for the given sequence of layers
 * \param dbn The DBN to build the context from
 */
template<template<typename, typename, size_t> typename Context, typename DBN, size_t... I>
auto build_context(DBN& dbn, std::index_sequence<I...> /*seq*/){
    return std::make_tuple
        (
            (std::make_pair(
                std::ref(dbn.template layer_get<I>()),  // Reference to the layer
                std::make_shared<Context<DBN, typename DBN::template layer_type<I>, I>>(dbn.template layer_get<I>()))
            )...
        );
}

/*!
 * \brief Build the context for a DBN
 * \param dbn The DBN to build the context from
 */
template<template<typename, typename, size_t> typename Context, typename DBN>
auto build_context(DBN& dbn){
    return build_context<Context>(dbn, std::make_index_sequence<DBN::layers>());
}

/*!
 * \brief Simple gradient descent trainer
 */
template <typename DBN>
struct sgd_trainer {
    using dbn_t     = DBN;                    ///< The type of DBN being trained
    using weight    = typename dbn_t::weight; ///< The data type for this layer
    using this_type = sgd_trainer<dbn_t>;     ///< The type of this layer

    static constexpr auto layers     = dbn_t::layers;     ///< The number of layers
    static constexpr auto batch_size = dbn_t::batch_size; ///< The batch size for training

    dbn_t& dbn;                                                  ///< The DBN being trained
    decltype(build_context<full_sgd_context>(dbn)) full_context; ///< The context
    size_t iteration;                                            ///< The current iteration

    // Transform layers need to inherit dimensions from back

    /*!
     * \brief Inherit the layer dimensions from front
     */
    template<typename L1, typename L2, cpp_enable_iff(decay_layer_traits<typename L2::first_type>::is_transform_layer())>
    static void inherit_from_front(L1& l1, L2& l2){
        auto& ctx1 = *l1.second;
        auto& ctx2 = *l2.second;

        if (ctx2.errors.size() == 0) {
            ctx2.output = ctx1.output;
            ctx2.errors = ctx1.output;
            ctx2.input  = ctx1.output;
        }
    }

    /*!
     * \brief Inherit the layer dimensions from front
     */
    template<typename L1, typename L2, cpp_disable_iff(decay_layer_traits<typename L2::first_type>::is_transform_layer())>
    static void inherit_from_front(L1& /*l1*/, L2& /*l2*/){ }

    /*!
     * \brief construct a new sgd_trainer
     * \param dbn The DBN being trained
     */
    explicit sgd_trainer(dbn_t& dbn) : dbn(dbn), full_context(build_context<full_sgd_context>(dbn)), iteration(1) {
        // Inherit dimensions from front to end (for transform layers)

        cpp::for_each_pair(full_context, [](auto& layer_ctx_1, auto& layer_ctx_2) {
            constexpr bool l2_transform = decay_layer_traits<decltype(layer_ctx_2.first)>::is_transform_layer();

            if (l2_transform) {
                this_type::inherit_from_front(layer_ctx_1, layer_ctx_2);
            }
        });
    }

    /*!
     * \brief Initialize the training
     */
    void init_training(size_t) {}

    // CPP17 Replace SFINAE with if constexpr

    /*!
     * \brief Compute the errors of the last layer given the loss function
     */
    template<loss_function F, typename Labels, cpp_enable_iff(F == loss_function::CATEGORICAL_CROSS_ENTROPY)>
    void last_errors(bool full_batch, size_t n, const Labels& labels){
        auto& last_ctx   = *std::get<layers - 1>(full_context).second;

        if (cpp_unlikely(!full_batch)) {
            last_ctx.errors = 0;

            for (size_t i = 0; i < n; ++i) {
                last_ctx.errors(i) = labels(i) - last_ctx.output(i);
            }
        } else {
            last_ctx.errors = labels - last_ctx.output;
        }

        // Note: No need to multiply by the derivative of
        // the activation function since the terms are
        // canceling out in the derivative of the loss
    }

    /*!
     * \brief Compute the errors of the last layer given the loss function
     */
    template<loss_function F, typename Labels, cpp_enable_iff(F == loss_function::MEAN_SQUARED_ERROR)>
    void last_errors(bool full_batch, size_t n, const Labels& labels){
        auto& last_layer = std::get<layers - 1>(full_context).first;
        auto& last_ctx   = *std::get<layers - 1>(full_context).second;

        if (cpp_unlikely(!full_batch)) {
            last_ctx.errors = 0;

            for (size_t i = 0; i < n; ++i) {
                last_ctx.errors(i) = 2.0 * (labels(i) - last_ctx.output(i));
            }
        } else {
            last_ctx.errors = 2.0 * (labels - last_ctx.output);
        }

        // Multiply by the derivative of the activation function
        last_layer.adapt_errors(last_ctx);
    }

    /*!
     * \brief Compute the errors of the last layer given the loss function
     */
    template<loss_function F, typename Labels, cpp_enable_iff(F == loss_function::BINARY_CROSS_ENTROPY)>
    void last_errors(bool full_batch, size_t n, const Labels& labels){
        auto& last_layer = std::get<layers - 1>(full_context).first;
        auto& last_ctx   = *std::get<layers - 1>(full_context).second;

        // Avoid Nan from division by ((1 - out) * out)
        auto out = etl::force_temporary(etl::clip(last_ctx.output, 0.001, 0.999));

        if (cpp_unlikely(!full_batch)) {
            last_ctx.errors = 0;

            for (size_t i = 0; i < n; ++i) {
                last_ctx.errors(i) = (labels(i) - out(i)) / ((1.0 - out(i)) >> out(i));
            }
        } else {
            last_ctx.errors = (labels - out) / ((1.0 - out) >> out);
        }

        // Check for NAN before derivative
        nan_check_etl(last_ctx.errors);

        // Multiply by the derivative of the activation function
        last_layer.adapt_errors(last_ctx);

        // Check for NAN after derivative
        nan_check_etl(last_ctx.errors);
    }

    /*!
     * \brief Train a batch of data
     * \param epoch The current epoch
     * \param inputs A batch of inputs
     * \param labels A batch of labels
     * \return a pair containing the error and the loss for the batch
     */
    template <bool Error, typename Inputs, typename Labels>
    std::pair<double, double> train_batch(size_t epoch, const Inputs& inputs, const Labels& labels) {
        dll::auto_timer timer("sgd::train_batch");

        auto& first_layer = std::get<0>(full_context).first;
        auto& first_ctx   = *std::get<0>(full_context).second;
        auto& last_ctx    = *std::get<layers - 1>(full_context).second;

        const auto n          = etl::dim<0>(inputs);
        const bool full_batch = n == etl::dim<0>(first_ctx.input);

        // Ensure that the data batch and the label batch are of the same size
        cpp_assert(n == etl::dim<0>(labels), "Invalid sizes");

        // Ensure that the context can hold the inputs
        cpp_assert(n <= etl::dim<0>(first_ctx.input), "Invalid sizes");

        //Feedforward pass

        {
            dll::auto_timer timer("sgd::forward");

            forward_batch_helper<true>(inputs);
        }

        {
            dll::auto_timer timer("sgd::backward");

            //Compute the errors of the last layer

            last_errors<dbn_t::loss>(full_batch, n, labels);

            // Backpropagate the error

            bool last = true;

            cpp::for_each_rpair(full_context, [&last](auto& layer_ctx_1, auto& layer_ctx_2) {
                backward_layer(layer_ctx_2.first, *layer_ctx_2.second, get_errors(*layer_ctx_1.second), last);
            });

            first_layer.adapt_errors(first_ctx);
        }

        // Compute and apply the gradients

        {
            dll::auto_timer timer("sgd::grad");

            cpp::for_each(full_context, [this, epoch, n](auto& layer_ctx) {
                this->apply_gradients_layer(epoch, n, layer_ctx.first, *layer_ctx.second);
            });
        }

        // Update the counter of iterations
        ++iteration;

        // Compute error and loss

        if constexpr (Error) {
            dll::auto_timer timer("sgd::error");

            auto[error, loss] = dbn.evaluate_metrics_batch(last_ctx.output, labels, n, true);

            return std::make_pair(error, loss);
        } else {
            return {0, 0};
        }
    }

    template <typename Layer, typename Context>
    void apply_gradients_layer(size_t epoch, size_t n, Layer& layer, Context& context){
        if constexpr (is_utility_layer<Layer>) {
            cpp::for_each(layer.layers, context.sub_contexts, [this, epoch, n](auto& sub_layer, auto& sub_context) {
                this->apply_gradients_layer(epoch, n, sub_layer, sub_context);
            });
        } else {
            // Compute the gradients
            layer.compute_gradients(context);

            // Apply the gradients
            this->update_weights<dbn_traits<dbn_t>::updater()>(epoch, layer, context, n);
        }
    }

    template <typename Layer, typename Context, typename Errors, cpp_disable_iff(is_utility_layer<Layer>)>
    static void backward_layer(Layer& layer, Context& context, Errors&& errors, bool& last){
        if(!last){
            layer.adapt_errors(context);
        }

        last = false;

        layer.backward_batch(errors, context);
    }

    template <size_t L, typename Layer, typename Context, typename Errors>
    static void backward_layer_group(Layer& layer, Context& context, Errors&& errors, bool& last){
        auto& sub_layer   = std::get<L>(layer.layers);
        auto& sub_context = std::get<L>(context.sub_contexts);

        if (!last) {
            sub_layer.adapt_errors(sub_context);
        }

        last = false;

        if constexpr (L == 0) {
            sub_layer.backward_batch(errors, sub_context);
        } else {
            auto& prev_sub_context = std::get<L - 1>(context.sub_contexts);

            sub_layer.backward_batch(prev_sub_context.errors, sub_context);

            backward_layer_group<L - 1>(layer, context, errors, last);
        }
    }

    template <typename Layer, typename Context, typename Errors, cpp_enable_iff(is_group_layer<Layer>)>
    static void backward_layer(Layer& layer, Context& context, Errors&& errors, bool& last){
        backward_layer_group<Layer::n_layers - 1>(layer, context, errors, last);
    }

    template <typename Layer, typename Context, typename Errors, cpp_enable_iff(is_merge_layer<Layer>)>
    static void backward_layer(Layer& layer, Context& context, Errors&& errors, bool& last){
        errors = 0;

        // Dispatch all the sub contexts

        cpp::for_each_i(layer.layers, context.sub_contexts, [&context, &errors, &last](size_t i, auto& sub_layer, auto& sub_context){
            batch_dispatch(get_errors(sub_context), context.errors, i);

            auto back_errors = errors;

            bool sub_last = last;
            backward_layer(sub_layer, sub_context, back_errors, sub_last);

            errors += back_errors;
        });

        last = false;
    }

    template <bool Train, typename Layer, typename Inputs, typename Context, cpp_disable_iff(is_utility_layer<Layer>)>
    static void forward_layer(Layer& layer, Inputs&& inputs, Context& context) {
        context.input = inputs;

        if constexpr (Train) {
            layer.train_forward_batch(context.output, context.input);
        } else {
            layer.test_forward_batch(context.output, context.input);
        }
    }

    template <bool Train, size_t L, typename Layer, typename Inputs, typename Context>
    static void forward_layer_group(Layer& layer, Inputs&& inputs, Context& context) {
        if constexpr (L < Layer::n_layers) {
            auto& sub_layer   = std::get<L>(layer.layers);
            auto& sub_context = std::get<L>(context.sub_contexts);

            sub_context.input = inputs;

            if constexpr (Train) {
                sub_layer.train_forward_batch(sub_context.output, sub_context.input);
            } else {
                sub_layer.test_forward_batch(sub_context.output, sub_context.input);
            }

            forward_layer_group<Train, L + 1>(layer, sub_context.output, context);
        }
    }

    template <bool Train, typename Layer, typename Inputs, typename Context, cpp_enable_iff(is_group_layer<Layer>)>
    static void forward_layer(Layer& layer, Inputs&& inputs, Context& context) {
        forward_layer_group<Train, 0>(layer, inputs, context);
    }

    template <bool Train, size_t L, typename Layer, typename Inputs, typename Context>
    static void forward_layer_merge(Layer& layer, Inputs&& inputs, Context& context) {
        if constexpr (L < Layer::n_layers) {
            auto& sub_layer   = std::get<L>(layer.layers);
            auto& sub_context = std::get<L>(context.sub_contexts);

            forward_layer<Train>(sub_layer, inputs, sub_context);

            forward_layer_merge<Train, L + 1>(layer, inputs, context);
        }
    }

    template <bool Train, typename Layer, typename Inputs, typename Context, cpp_enable_iff(is_merge_layer<Layer>)>
    static void forward_layer(Layer& layer, Inputs&& inputs, Context& context) {
        context.input = inputs;

        // Fully forward each group

        forward_layer_merge<Train, 0>(layer, context.input, context);

        // Concatenate all the sub contexts

        cpp::for_each_i(context.sub_contexts, [&context](size_t i, auto& sub_context){
            batch_merge(context.output, get_output(sub_context), i);
        });
    }

    template <typename Context>
    static auto& get_output(Context& context) {
        if constexpr (is_group_layer<typename Context::layer_t>) {
            return get_output(std::get<Context::n_layers - 1>(context.sub_contexts));
        } else {
            return context.output;
        }
    }

    template <typename Context>
    static auto& get_errors(Context& context) {
        if constexpr (is_group_layer < typename Context::layer_t>) {
            return get_errors(std::get<Context::n_layers - 1>(context.sub_contexts));
        } else {
            return context.errors;
        }
    }

    //TODO
    template <bool Train, typename Inputs>
    auto& forward_batch_helper([[maybe_unused]] dbn_t& dbn, Inputs&& inputs) {
        return this->template forward_batch_helper<Train>(inputs);
    }

    template <bool Train, typename Inputs>
    auto& forward_batch_helper(Inputs&& inputs) {
        auto& first_layer = std::get<0>(full_context).first;
        auto& first_ctx   = *std::get<0>(full_context).second;
        auto& last_ctx    = *std::get<layers - 1>(full_context).second;

        const auto n          = etl::dim<0>(inputs);
        const bool full_batch = n == etl::dim<0>(first_ctx.input);

        // Ensure that the context can hold the inputs
        cpp_assert(n <= etl::dim<0>(first_ctx.input), "Invalid sizes");

        if (cpp_unlikely(!full_batch)) {
            first_ctx.input  = 0;
            first_ctx.output = 0;

            for (size_t i = 0; i < etl::dim<0>(inputs); ++i) {
                first_ctx.input(i) = inputs(i);
            }
        } else {
            first_ctx.input = inputs;
        }

        if constexpr (Train) {
            first_layer.train_forward_batch(first_ctx.output, first_ctx.input);
        } else {
            first_layer.test_forward_batch(first_ctx.output, first_ctx.input);
        }

        cpp::for_each_pair(full_context, [this](auto& layer_ctx_1, auto& layer_ctx_2) {
            this->template forward_layer<Train>(layer_ctx_2.first, get_output(*layer_ctx_1.second), *layer_ctx_2.second);
        });

        return last_ctx.output;
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C>
    void update_weights([[maybe_unused]] size_t epoch, [[maybe_unused]] L& layer, [[maybe_unused]] C& context, [[maybe_unused]] size_t n) {
        if constexpr (decay_layer_traits<L>::is_neural_layer()) {
            dll::auto_timer timer("sgd::update_weights");

            // Update all variables of the layer

            static constexpr size_t N = std::tuple_size<decltype(layer.trainable_parameters())>();

            update_variables<UT>(epoch, layer, context, n, std::make_index_sequence<N>());
        }
    }

    template <updater_type UT, typename L, typename C, size_t... I>
    void update_variables(size_t epoch, L& layer, C& context, size_t n, std::index_sequence<I...> /* args */) {
        (update_variable<I, UT>(epoch, layer, context, n), ...);
    }

    template <size_t I, updater_type UT, typename L, typename C>
    void update_variable(size_t epoch, L& layer, C& context, size_t n) {
        // 1. Decay the learning rate (if necessary)

        auto eps             = dbn.learning_rate;
        const auto eps_decay = dbn.learning_rate_decay;

        if (eps_decay > 0.0) {
            eps *= 1.0 / (1.0 + eps_decay * iteration);
        }

        //2. Update the gradients (L1/L2 and gradient clipping)

        auto& w      = std::get<I>(layer.trainable_parameters());
        auto& w_grad = std::get<I>(context.up.context)->grad;

        // Note the distinction for w and b for decay is far from optimal...
        if constexpr (I == 0) {
            this->update_grad<w_decay(dbn_traits<dbn_t>::decay())>(w, w_grad, n);
        } else {
            this->update_grad<b_decay(dbn_traits<dbn_t>::decay())>(w, w_grad, n);
        }

        // 3. Apply the gradients

        apply_gradients<I, UT>(epoch, layer, context, n, eps);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <size_t I, updater_type UT, typename L, typename C, cpp_enable_iff(UT == updater_type::SGD)>
    void apply_gradients([[maybe_unused]] size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:sgd");

        auto& w      = std::get<I>(layer.trainable_parameters());
        auto& w_grad = std::get<I>(context.up.context)->grad;

        w += (eps / n) * w_grad;

        nan_check_deep(w);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <size_t I, updater_type UT, typename L, typename C, cpp_enable_iff(UT == updater_type::MOMENTUM)>
    void apply_gradients([[maybe_unused]] size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:momentum");

        const auto momentum = dbn.momentum;

        auto& w      = std::get<I>(layer.trainable_parameters());
        auto& w_grad = std::get<I>(context.up.context)->grad;
        auto& w_inc  = std::get<I>(context.up.context)->inc;

        //Update with momentum and learning rate

        w_inc = momentum * w_inc + (eps / n) * w_grad;

        w += w_inc;

        nan_check_deep(w);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <size_t I, updater_type UT, typename L, typename C, cpp_enable_iff(UT == updater_type::NESTEROV)>
    void apply_gradients([[maybe_unused]] size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:nesterov");

        const auto momentum = dbn.momentum;

        auto& w          = std::get<I>(layer.trainable_parameters());
        auto& w_grad     = std::get<I>(context.up.context)->grad;
        auto& w_inc      = std::get<I>(context.up.context)->inc;
        auto& w_inc_prev = std::get<I>(context.up.context)->inc_prev;

        //Update with momentum and learning rate

        w_inc_prev = w_inc;

        w_inc = momentum * w_inc + (eps / n) * w_grad;

        w += -momentum * w_inc_prev + (1.0 + momentum) * w_inc;

        nan_check_deep(w);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <size_t I, updater_type UT, typename L, typename C, cpp_enable_iff(UT == updater_type::ADAGRAD)>
    void apply_gradients([[maybe_unused]] size_t epoch, L& layer, C& context, [[maybe_unused]] size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:adagrad");

        const auto e = 1e-8;

        auto& w      = std::get<I>(layer.trainable_parameters());
        auto& w_grad = std::get<I>(context.up.context)->grad;
        auto& w_inc  = std::get<I>(context.up.context)->inc;

        w_inc = w_inc + (w_grad >> w_grad);

        w += (eps * w_grad) / etl::sqrt(w_inc + e);

        nan_check_deep(w);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <size_t I, updater_type UT, typename L, typename C, cpp_enable_iff(UT == updater_type::ADADELTA)>
    void apply_gradients([[maybe_unused]] size_t epoch, L& layer, C& context, [[maybe_unused]] size_t n, [[maybe_unused]] weight eps) {
        dll::auto_timer timer("sgd::apply_grad:adadelta");

        const auto beta = dbn.adadelta_beta;
        const auto e = 1e-8;

        auto& w      = std::get<I>(layer.trainable_parameters());
        auto& w_grad = std::get<I>(context.up.context)->grad;
        auto& w_g    = std::get<I>(context.up.context)->g;
        auto& w_v    = std::get<I>(context.up.context)->v;
        auto& w_x    = std::get<I>(context.up.context)->x;

        w_g = beta * w_g + ((1.0 - beta) * (w_grad >> w_grad));
        w_v = (sqrt(w_x + e) >> w_grad) / sqrt(w_g + e);
        w_x = beta * w_x + ((1.0 - beta) * (w_v >> w_v));

        w += w_v;

        nan_check_deep(w);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <size_t I, updater_type UT, typename L, typename C, cpp_enable_iff(UT == updater_type::ADAM)>
    void apply_gradients([[maybe_unused]] size_t epoch, L& layer, C& context, [[maybe_unused]] size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:adam");

        const auto beta1 = dbn.adam_beta1;
        const auto beta2 = dbn.adam_beta2;
        const auto e = 1e-8;

        auto& w      = std::get<I>(layer.trainable_parameters());
        auto& w_grad = std::get<I>(context.up.context)->grad;
        auto& w_m    = std::get<I>(context.up.context)->m;
        auto& w_v    = std::get<I>(context.up.context)->v;

        // Standard Adam estimations of the first and second moments

        w_m = beta1 * w_m + ((1.0 - beta1) * w_grad);
        w_v = beta2 * w_v + ((1.0 - beta2) * (w_grad >> w_grad));

        // Update the parameters

        w += (eps * w_m) / (etl::sqrt(w_v) + e);

        nan_check_deep(w);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <size_t I, updater_type UT, typename L, typename C, cpp_enable_iff(UT == updater_type::ADAM_CORRECT)>
    void apply_gradients([[maybe_unused]] size_t epoch, L& layer, C& context, [[maybe_unused]] size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:adam_correct");

        const auto beta1 = dbn.adam_beta1;
        const auto beta2 = dbn.adam_beta2;
        const auto e = 1e-8;
        const auto t = iteration;

        auto& w      = std::get<I>(layer.trainable_parameters());
        auto& w_grad = std::get<I>(context.up.context)->grad;
        auto& w_m    = std::get<I>(context.up.context)->m;
        auto& w_mt   = std::get<I>(context.up.context)->mt;
        auto& w_v    = std::get<I>(context.up.context)->v;
        auto& w_vt   = std::get<I>(context.up.context)->vt;

        // Standard Adam estimations of the first and second moments

        w_m = beta1 * w_m + ((1.0 - beta1) * w_grad);
        w_v = beta2 * w_v + ((1.0 - beta2) * (w_grad >> w_grad));

        // Correct the bias (towards zero) of the first and second moments

        w_mt = w_m / (1.0 - std::pow(beta1, t));
        w_vt = w_v / (1.0 - std::pow(beta2, t));

        // Update the parameters

        w += (eps * w_m) / (etl::sqrt(w_v) + e);

        nan_check_deep(w);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <size_t I, updater_type UT, typename L, typename C, cpp_enable_iff(UT == updater_type::ADAMAX)>
    void apply_gradients([[maybe_unused]] size_t epoch, L& layer, C& context, [[maybe_unused]] size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:adamax");

        const auto beta1 = dbn.adam_beta1;
        const auto beta2 = dbn.adam_beta2;

        auto& w      = std::get<I>(layer.trainable_parameters());
        auto& w_grad = std::get<I>(context.up.context)->grad;
        auto& w_m    = std::get<I>(context.up.context)->m;
        auto& w_v    = std::get<I>(context.up.context)->v;

        // Standard Adam estimations of the first moment

        w_m = beta1 * w_m + ((1.0 - beta1) >> w_grad);

        // Estimation of the second moment with infinite-norm

        w_v = etl::max(beta2 * w_v, etl::abs(w_grad));

        // Update the parameters

        w += (eps >> w_m) / w_v;

        nan_check_deep(w);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <size_t I, updater_type UT, typename L, typename C, cpp_enable_iff(UT == updater_type::NADAM)>
    void apply_gradients([[maybe_unused]] size_t epoch, L& layer, C& context, [[maybe_unused]] size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:nadam");

        // The scalar parameters
        const weight beta1          = dbn.adam_beta1;
        const weight beta2          = dbn.adam_beta2;
        const weight schedule_decay = dbn.nadam_schedule_decay;
        const weight e              = 1e-8;
        const weight t              = iteration;

        auto& w          = std::get<I>(layer.trainable_parameters());
        auto& w_grad     = std::get<I>(context.up.context)->grad;
        auto& w_m        = std::get<I>(context.up.context)->m;
        auto& w_v        = std::get<I>(context.up.context)->v;
        auto& m_schedule = std::get<I>(context.up.context)->m_schedule;

        // Compute the schedule for momentum

        weight momentum_cache_t   = beta1 * (1.0 - 0.5 * (std::pow(0.96, t * schedule_decay)));
        weight momentum_cache_t_1 = beta1 * (1.0 - 0.5 * (std::pow(0.96, (t + 1) * schedule_decay)));

        weight m_schedule_new  = m_schedule * momentum_cache_t;
        weight m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1;

        if constexpr (I == 0) {
            m_schedule = m_schedule_new;
        }

        // Standard Adam estimations of the first and second order moments

        w_m = beta1 * w_m + ((weight(1) - beta1) >> w_grad);
        w_v = beta2 * w_v + ((weight(1) - beta2) >> (w_grad >> w_grad));

        // Correct the bias (towards zero) of the first and second moments

        // Inlined into the final expression for performance
        //w_mt = w_m / (1.0 - m_schedule_next);
        //w_vt = w_v / (1.0 - std::pow(beta2, t));

        // Update the parameters

        weight f1 = 1.0 - momentum_cache_t;
        weight f2 = 1.0 - m_schedule_new;

        weight m1 = eps * (f1 / f2);
        weight m2 = eps * momentum_cache_t_1;

        // Compute the new weights
        // Basic version: w += (m1 * w_grad + m2 * w_mt) / (etl::sqrt(w_vt) + e);
        // Optimized for performance into:
        //w += (m1 * w_grad + m2 * (w_m / (weight(1) - m_schedule_next))) / (etl::sqrt(w_v / (weight(1) - std::pow(beta2, t))) + e);
        // Optimized into
        w += (m1 * w_grad + (m2 / (weight(1) - m_schedule_next)) * w_m) / (etl::sqrt(w_v / (weight(1) - std::pow(beta2, t))) + e);

        nan_check_deep(w);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <size_t I, updater_type UT, typename L, typename C, cpp_enable_iff(UT == updater_type::RMSPROP)>
    void apply_gradients([[maybe_unused]] size_t epoch, L& layer, C& context, [[maybe_unused]] size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:rmsprop");

        const auto decay = dbn.rmsprop_decay;
        const auto e = 1e-8;

        auto& w      = std::get<I>(layer.trainable_parameters());
        auto& w_grad = std::get<I>(context.up.context)->grad;
        auto& w_inc  = std::get<I>(context.up.context)->inc;

        w_inc = decay * w_inc + (1 - decay) * (w_grad >> w_grad);

        w += (eps >> w_grad) / etl::sqrt(w_inc + e);

        nan_check_deep(w);
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <typename G>
    void clip_gradients(G& grad, size_t n) {
        if constexpr (dbn_traits<dbn_t>::has_clip_gradients()) {
            const auto t            = dbn.gradient_clip;
            const auto grad_l2_norm = std::sqrt(etl::sum(grad >> grad) / (n * n));

            if (grad_l2_norm > t) {
                grad = grad >> (t / grad_l2_norm);
            }
        }
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <decay_type decay, typename V, typename G>
    void update_grad(const V& value, G& grad, size_t n) {
        if constexpr (decay == decay_type::L1) {
            grad = grad - dbn.l1_weight_cost * abs(value);
        } else if constexpr (decay == decay_type::L2) {
            grad = grad - dbn.l2_weight_cost * value;
        } else if constexpr (decay == decay_type::L1L2) {
            grad = grad - dbn.l1_weight_cost * abs(value) - dbn.l2_weight_cost * value;
        }

        clip_gradients(grad, n);
    }

    /*!
     * \brief Return the name of the trainer
     */
    static std::string name() {
        return "Stochastic Gradient Descent";
    }
};

} //end of dll namespace
