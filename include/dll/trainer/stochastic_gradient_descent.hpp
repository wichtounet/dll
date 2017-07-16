//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

#include "cpp_utils/static_if.hpp"
#include "cpp_utils/tuple_utils.hpp"

#include "dll/util/checks.hpp" // For NaN checks
#include "dll/util/timers.hpp" // For auto_timer
#include "dll/context.hpp"     // For sgd_context
#include "dll/trainer/context_fwd.hpp" // For sgd_context

namespace dll {

/*!
 * \brief The context for the updater
 */
template <updater_type UT, bool Neural, typename Context>
struct updater_context {
    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context(Context& context) {
        cpp_unused(context);
    }
};

/*!
 * \brief The context for the Momentum updater
 */
template <typename Context>
struct updater_context<updater_type::MOMENTUM, true, Context> {
    using w_grad_t = decltype(std::declval<Context>().w_grad); ///< The for the weight gradients
    using b_grad_t = decltype(std::declval<Context>().b_grad); ///< The for the weight biases

    w_grad_t w_inc; //< The momentum cache for the weights
    b_grad_t b_inc; //< The momentum cache for the biases

    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context(Context& context) : w_inc(context.w_grad), b_inc(context.b_grad) {
        w_inc = 0;
        b_inc = 0;
    }
};

/*!
 * \brief The context for the Nesterov Momentum updater
 */
template <typename Context>
struct updater_context<updater_type::NESTEROV, true, Context> {
    using w_grad_t = decltype(std::declval<Context>().w_grad); ///< The for the weight gradients
    using b_grad_t = decltype(std::declval<Context>().b_grad); ///< The for the weight biases

    w_grad_t w_inc; //< The momentum cache for the weights
    b_grad_t b_inc; //< The momentum cache for the biases
    w_grad_t w_inc_prev; //< The momentum cache for the weights
    b_grad_t b_inc_prev; //< The momentum cache for the biases

    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context(Context& context) : w_inc(context.w_grad), b_inc(context.b_grad), w_inc_prev(context.w_grad), b_inc_prev(context.b_grad) {
        w_inc = 0;
        b_inc = 0;
    }
};

/*!
 * \brief The context for the RMSPROP updater
 */
template <typename Context>
struct updater_context<updater_type::RMSPROP, true, Context> {
    using w_grad_t = decltype(std::declval<Context>().w_grad); ///< The for the weight gradients
    using b_grad_t = decltype(std::declval<Context>().b_grad); ///< The for the weight biases

    w_grad_t w_inc; //< The rmsprop cache for the weights
    b_grad_t b_inc; //< The rmsprop cache for the biases

    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context(Context& context) : w_inc(context.w_grad), b_inc(context.b_grad) {
        w_inc = 0;
        b_inc = 0;
    }
};

/*!
 * \brief The context for the Adagrad updater
 */
template <typename Context>
struct updater_context<updater_type::ADAGRAD, true, Context> {
    using w_grad_t = decltype(std::declval<Context>().w_grad); ///< The for the weight gradients
    using b_grad_t = decltype(std::declval<Context>().b_grad); ///< The for the weight biases

    w_grad_t w_inc; //< The adagrad cache for the weights
    b_grad_t b_inc; //< The adagrad cache for the biases

    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context(Context& context) : w_inc(context.w_grad), b_inc(context.b_grad) {
        w_inc = 0;
        b_inc = 0;
    }
};

/*!
 * \brief The context for the Adadelta updater
 */
template <typename Context>
struct updater_context<updater_type::ADADELTA, true, Context> {
    using w_grad_t = decltype(std::declval<Context>().w_grad); ///< The for the weight gradients
    using b_grad_t = decltype(std::declval<Context>().b_grad); ///< The for the weight biases

    w_grad_t w_g; //< The Adadelta cache for the weights
    w_grad_t w_x; //< The Adadelta cache for the weights
    w_grad_t w_v; //< The Adadelta cache for the weights

    b_grad_t b_g; //< The Adadelta cache for the biases
    b_grad_t b_x; //< The Adadelta cache for the biases
    b_grad_t b_v; //< The Adadelta cache for the biases

    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context(Context& context) : w_g(context.w_grad), w_x(context.w_grad), w_v(context.w_grad), b_g(context.b_grad), b_x(context.b_grad), b_v(context.b_grad) {
        w_g = 0;
        w_x = 0;
        w_v = 0;

        b_g = 0;
        b_x = 0;
        b_v = 0;
    }
};

/*!
 * \brief The context for the Adam updater
 */
template <typename Context>
struct updater_context<updater_type::ADAM, true, Context> {
    using w_grad_t = decltype(std::declval<Context>().w_grad); ///< The for the weight gradients
    using b_grad_t = decltype(std::declval<Context>().b_grad); ///< The for the weight biases

    w_grad_t w_m; //< The Adam cache for the weights
    w_grad_t w_v; //< The Adam cache for the weights
    b_grad_t b_m; //< The Adam cache for the biases
    b_grad_t b_v; //< The Adam cache for the biases

    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context(Context& context) : w_m(context.w_grad), w_v(context.w_grad), b_m(context.b_grad), b_v(context.b_grad) {
        w_m = 0;
        w_v = 0;
        b_m = 0;
        b_v = 0;
    }
};

/*!
 * \brief The context for the Adam with bias correction updater
 */
template <typename Context>
struct updater_context<updater_type::ADAM_CORRECT, true, Context> {
    using w_grad_t = decltype(std::declval<Context>().w_grad); ///< The for the weight gradients
    using b_grad_t = decltype(std::declval<Context>().b_grad); ///< The for the weight biases

    w_grad_t w_m; //< The Adam cache for the weights
    w_grad_t w_v; //< The Adam cache for the weights
    b_grad_t b_m; //< The Adam cache for the biases
    b_grad_t b_v; //< The Adam cache for the biases

    w_grad_t w_mt; //< The Adam cache for the weights
    w_grad_t w_vt; //< The Adam cache for the weights
    b_grad_t b_mt; //< The Adam cache for the biases
    b_grad_t b_vt; //< The Adam cache for the biases

    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context(Context& context) : w_m(context.w_grad), w_v(context.w_grad), b_m(context.b_grad), b_v(context.b_grad), w_mt(context.w_grad), w_vt(context.w_grad), b_mt(context.b_grad), b_vt(context.b_grad) {
        w_m = 0;
        w_v = 0;
        b_m = 0;
        b_v = 0;

        w_mt = 0;
        w_vt = 0;
        b_mt = 0;
        b_vt = 0;
    }
};

/*!
 * \brief The context for the Adamax updater
 */
template <typename Context>
struct updater_context<updater_type::ADAMAX, true, Context> {
    using w_grad_t = decltype(std::declval<Context>().w_grad); ///< The for the weight gradients
    using b_grad_t = decltype(std::declval<Context>().b_grad); ///< The for the weight biases

    w_grad_t w_m; //< The Adam cache for the weights
    w_grad_t w_v; //< The Adam cache for the weights
    b_grad_t b_m; //< The Adam cache for the biases
    b_grad_t b_v; //< The Adam cache for the biases

    /*!
     * \brief Construct a new updater_context using the parent context
     */
    updater_context(Context& context) : w_m(context.w_grad), w_v(context.w_grad), b_m(context.b_grad), b_v(context.b_grad) {
        w_m = 0;
        w_v = 0;
        b_m = 0;
        b_v = 0;
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
    updater_context<DBN::updater, decay_layer_traits<Layer>::is_neural_layer(), context_type> up;

    /*!
     * \brief Construct the full_sgd_context for the given layer
     */
    full_sgd_context(Layer& layer) : context_type(layer), up(*this) {
        // Nothing else to init
    }
};

/*!
 * \brief Build the context for a DBN for the given sequence of layers
 * \param dbn The DBN to build the context from
 */
template<template<typename, typename, size_t> class Context, typename DBN, size_t... I>
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
template<template<typename, typename, size_t> class Context, typename DBN>
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
    size_t iteration;

    // Transform layers need to inherit dimensions from back

    /*!
     * \brief Inherit the layer dimensions from front
     */
    template<typename L1, typename L2, cpp_enable_if(decay_layer_traits<typename L2::first_type>::is_transform_layer())>
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
    template<typename L1, typename L2, cpp_disable_if(decay_layer_traits<typename L2::first_type>::is_transform_layer())>
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
    template<loss_function F, typename Labels, cpp_enable_if(F == loss_function::CATEGORICAL_CROSS_ENTROPY)>
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
    template<loss_function F, typename Labels, cpp_enable_if(F == loss_function::MEAN_SQUARED_ERROR)>
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
    template<loss_function F, typename Labels, cpp_enable_if(F == loss_function::BINARY_CROSS_ENTROPY)>
    void last_errors(bool full_batch, size_t n, const Labels& labels){
        auto& last_layer = std::get<layers - 1>(full_context).first;
        auto& last_ctx   = *std::get<layers - 1>(full_context).second;

        if (cpp_unlikely(!full_batch)) {
            last_ctx.errors = 0;

            auto& out = last_ctx.output;

            for (size_t i = 0; i < n; ++i) {
                last_ctx.errors(i) = (labels(i) - out(i)) / ((1.0 - out(i)) >> out(i));
            }
        } else {
            auto& out = last_ctx.output;

            last_ctx.errors = (labels - out) / ((1.0 - out) >> out);
        }

        // Multiply by the derivative of the activation function
        last_layer.adapt_errors(last_ctx);
    }

    /*!
     * \brief Train a batch of data
     * \param epoch The current epoch
     * \param inputs A batch of inputs
     * \param labels A batch of labels
     * \return a pair containing the error and the loss for the batch
     */
    template <typename Inputs, typename Labels>
    std::pair<double, double> train_batch(size_t epoch, const Inputs& inputs, const Labels& labels) {
        dll::auto_timer timer("sgd::train_batch");

        cpp_unused(epoch);

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

            if(cpp_unlikely(!full_batch)){
                first_ctx.input  = 0;
                first_ctx.output = 0;

                for (size_t i = 0; i < etl::dim<0>(inputs); ++i) {
                    first_ctx.input(i) = inputs(i);
                }
            } else {
                first_ctx.input = inputs;
            }

            first_layer.train_batch_activate_hidden(first_ctx.output, first_ctx.input);

            cpp::for_each_pair(full_context, [](auto& layer_ctx_1, auto& layer_ctx_2) {
                auto& layer_2 = layer_ctx_2.first;

                auto& ctx1 = *layer_ctx_1.second;
                auto& ctx2 = *layer_ctx_2.second;

                ctx2.input = ctx1.output;
                layer_2.train_batch_activate_hidden(ctx2.output, ctx2.input);
            });
        }

        {
            dll::auto_timer timer("sgd::backward");

            //Compute the errors of the last layer

            last_errors<dbn_t::loss>(full_batch, n, labels);

            // Backpropagate the error

            bool last = true;

            cpp::for_each_rpair(full_context, [&last](auto& layer_ctx_1, auto& layer_ctx_2) {
                auto& r2 = layer_ctx_2.first;

                auto& ctx1 = *layer_ctx_1.second;
                auto& ctx2 = *layer_ctx_2.second;

                if(!last){
                    r2.adapt_errors(ctx2);
                }

                last = false;

                r2.backward_batch(ctx1.errors, ctx2);
            });

            first_layer.adapt_errors(first_ctx);
        }

        // Compute and apply the gradients

        {
            dll::auto_timer timer("sgd::grad");

            cpp::for_each(full_context, [this, epoch, n](auto& layer_ctx) {
                // Compute the gradients
                layer_ctx.first.compute_gradients(*layer_ctx.second);

                // Apply the gradients
                this->update_weights<dbn_traits<dbn_t>::updater()>(epoch, layer_ctx.first, *layer_ctx.second, n);
            });
        }

        // Update the counter of iterations
        ++iteration;

        // Compute error and loss

        double error = 0.0;
        double loss = 0.0;

        {
            dll::auto_timer timer("sgd::error");

            std::tie(error, loss) = dbn.evaluate_metrics_batch(last_ctx.output, labels, n, true);
        }

        return std::make_pair(error, loss);
    }

    // CPP17 Replace with if constexpr

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(decay_layer_traits<L>::is_neural_layer())>
    void update_weights(size_t epoch, L& layer, C& context, size_t n) {
        dll::auto_timer timer("sgd::update_weights");

        //1. Update the gradients (L1/L2 and gradient clipping)

        this->update_grad<w_decay(dbn_traits<dbn_t>::decay())>(layer.w, context.w_grad, n);
        this->update_grad<b_decay(dbn_traits<dbn_t>::decay())>(layer.b, context.b_grad, n);

        // 2. Decay the learning rate

        auto eps             = dbn.learning_rate;
        const auto eps_decay = dbn.learning_rate_decay;

        if (eps_decay > 0.0) {
            eps *= 1.0 / (1.0 + eps_decay * iteration);
        }

        // 3. Apply the learning rate

        apply_gradients<UT>(epoch, layer, context, n, eps);
    }

    template <updater_type UT, typename L, typename C, cpp_disable_if(decay_layer_traits<L>::is_neural_layer())>
    void update_weights(size_t epoch, L& layer, C& context, size_t n) {
        cpp_unused(epoch);
        cpp_unused(layer);
        cpp_unused(context);
        cpp_unused(n);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(UT == updater_type::SGD)>
    void apply_gradients(size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:sgd");

        layer.w += (eps / n) * context.w_grad;
        layer.b += (eps / n) * context.b_grad;

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(epoch);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(UT == updater_type::MOMENTUM)>
    void apply_gradients(size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:momentum");

        //Update with momentum and learning rate
        auto momentum = dbn.momentum;

        context.up.w_inc = momentum * context.up.w_inc + (eps / n) * context.w_grad;
        context.up.b_inc = momentum * context.up.b_inc + (eps / n) * context.b_grad;

        layer.w += context.up.w_inc;
        layer.b += context.up.b_inc;

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(epoch);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(UT == updater_type::NESTEROV)>
    void apply_gradients(size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:nesterov");

        //Update with momentum and learning rate
        auto momentum = dbn.momentum;

        context.up.w_inc_prev = context.up.w_inc;
        context.up.b_inc_prev = context.up.b_inc;

        context.up.w_inc = momentum * context.up.w_inc + (eps / n) * context.w_grad;
        context.up.b_inc = momentum * context.up.b_inc + (eps / n) * context.b_grad;

        layer.w += (-momentum) * context.up.w_inc_prev + (1.0 + momentum) * context.up.w_inc;
        layer.b += (-momentum) * context.up.b_inc_prev + (1.0 + momentum) * context.up.b_inc;

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(epoch);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(UT == updater_type::ADAGRAD)>
    void apply_gradients(size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:adagrad");

        const auto e = 1e-8;

        context.up.w_inc = context.up.w_inc + (context.w_grad >> context.w_grad);
        context.up.b_inc = context.up.b_inc + (context.b_grad >> context.b_grad);

        layer.w += (eps >> context.w_grad) / etl::sqrt(context.up.w_inc + e);
        layer.b += (eps >> context.b_grad) / etl::sqrt(context.up.b_inc + e);

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(n);
        cpp_unused(epoch);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(UT == updater_type::ADADELTA)>
    void apply_gradients(size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:adadelta");

        const auto beta = dbn.adadelta_beta;
        const auto e = 1e-8;

        context.up.w_g = beta * context.up.w_g + ((1.0 - beta) >> context.w_grad >> context.w_grad);
        context.up.b_g = beta * context.up.b_g + ((1.0 - beta) >> context.b_grad >> context.b_grad);

        context.up.w_v = (sqrt(context.up.w_x + e) >> context.w_grad) / sqrt(context.up.w_g + e);
        context.up.b_v = (sqrt(context.up.b_x + e) >> context.b_grad) / sqrt(context.up.b_g + e);

        context.up.w_x = beta * context.up.w_x + ((1.0 - beta) >> context.up.w_v >> context.up.w_v);
        context.up.b_x = beta * context.up.b_x + ((1.0 - beta) >> context.up.b_v >> context.up.b_v);

        layer.w += context.up.w_v;
        layer.b += context.up.b_v;

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(n);
        cpp_unused(epoch);
        cpp_unused(eps);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(UT == updater_type::ADAM)>
    void apply_gradients(size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:adam");

        const auto beta1 = dbn.adam_beta1;
        const auto beta2 = dbn.adam_beta2;
        const auto e = 1e-8;

        // Standard Adam estimations of the first and second moments

        context.up.w_m = beta1 * context.up.w_m + ((1.0 - beta1) >> context.w_grad);
        context.up.b_m = beta1 * context.up.b_m + ((1.0 - beta1) >> context.b_grad);

        context.up.w_v = beta2 * context.up.w_v + ((1.0 - beta2) >> (context.w_grad >> context.w_grad));
        context.up.b_v = beta2 * context.up.b_v + ((1.0 - beta2) >> (context.b_grad >> context.b_grad));

        // Update the parameters

        layer.w += (eps >> context.up.w_m) / (etl::sqrt(context.up.w_v) + e);
        layer.b += (eps >> context.up.b_m) / (etl::sqrt(context.up.b_v) + e);

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(n);
        cpp_unused(epoch);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(UT == updater_type::ADAM_CORRECT)>
    void apply_gradients(size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:adam_correct");

        const auto beta1 = dbn.adam_beta1;
        const auto beta2 = dbn.adam_beta2;
        const auto e = 1e-8;
        const auto t = iteration;

        // Standard Adam estimations of the first and second moments

        context.up.w_m = beta1 * context.up.w_m + ((1.0 - beta1) >> context.w_grad);
        context.up.b_m = beta1 * context.up.b_m + ((1.0 - beta1) >> context.b_grad);

        context.up.w_v = beta2 * context.up.w_v + ((1.0 - beta2) >> (context.w_grad >> context.w_grad));
        context.up.b_v = beta2 * context.up.b_v + ((1.0 - beta2) >> (context.b_grad >> context.b_grad));

        // Correct the bias (towards zero) of the first and second moments

        context.up.w_mt = context.up.w_m / (1.0 - std::pow(beta1, t));
        context.up.b_mt = context.up.b_m / (1.0 - std::pow(beta1, t));

        context.up.w_vt = context.up.w_v / (1.0 - std::pow(beta2, t));
        context.up.b_vt = context.up.b_v / (1.0 - std::pow(beta2, t));

        // Update the parameters

        layer.w += (eps >> context.up.w_m) / (etl::sqrt(context.up.w_v) + e);
        layer.b += (eps >> context.up.b_m) / (etl::sqrt(context.up.b_v) + e);

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(n);
        cpp_unused(epoch);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(UT == updater_type::ADAMAX)>
    void apply_gradients(size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:adamax");

        const auto beta1 = dbn.adam_beta1;
        const auto beta2 = dbn.adam_beta2;

        // Standard Adam estimations of the first moment

        context.up.w_m = beta1 * context.up.w_m + ((1.0 - beta1) >> context.w_grad);
        context.up.b_m = beta1 * context.up.b_m + ((1.0 - beta1) >> context.b_grad);

        // Estimation of the second moment with infinite-norm

        context.up.w_v = etl::max(beta2 * context.up.w_v, etl::abs(context.w_grad));
        context.up.b_v = etl::max(beta2 * context.up.b_v, etl::abs(context.b_grad));

        // Update the parameters

        layer.w += (eps >> context.up.w_m) / context.up.w_v;
        layer.b += (eps >> context.up.b_m) / context.up.b_v;

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(n);
        cpp_unused(epoch);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(UT == updater_type::RMSPROP)>
    void apply_gradients(size_t epoch, L& layer, C& context, size_t n, weight eps) {
        dll::auto_timer timer("sgd::apply_grad:rmsprop");

        const auto decay = dbn.rmsprop_decay;
        const auto e = 1e-8;

        context.up.w_inc = decay * context.up.w_inc + (1 - decay) * (context.w_grad >> context.w_grad);
        context.up.b_inc = decay * context.up.b_inc + (1 - decay) * (context.b_grad >> context.b_grad);

        layer.w += (eps >> context.w_grad) / etl::sqrt(context.up.w_inc + e);
        layer.b += (eps >> context.b_grad) / etl::sqrt(context.up.b_inc + e);

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(n);
        cpp_unused(epoch);
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <typename D = dbn_t, typename G, cpp_enable_if(dbn_traits<D>::has_clip_gradients())>
    void clip_gradients(G& grad, size_t n) {
        const auto t = dbn.gradient_clip;
        const auto grad_l2_norm = std::sqrt(etl::sum(grad >> grad) / (n * n));

        if(grad_l2_norm > t){
            grad = grad >> (t / grad_l2_norm);
        }
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <typename D = dbn_t, typename G, cpp_disable_if(dbn_traits<D>::has_clip_gradients())>
    void clip_gradients(G& grad, size_t n) {
        cpp_unused(grad);
        cpp_unused(n);
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <decay_type decay, typename V, typename G, cpp_enable_if(decay == decay_type::L1)>
    void update_grad(const V& value, G& grad, size_t n) {
        grad = grad - dbn.l1_weight_cost * abs(value);

        clip_gradients(grad, n);
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <decay_type decay, typename V, typename G, cpp_enable_if(decay == decay_type::L2)>
    void update_grad(const V& value, G& grad, size_t n) {
        grad = grad - dbn.l2_weight_cost * value;

        clip_gradients(grad, n);
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <decay_type decay, typename V, typename G, cpp_enable_if(decay == decay_type::L1L2)>
    void update_grad(const V& value, G& grad, size_t n) {
        grad = grad - dbn.l1_weight_cost * abs(value) - dbn.l2_weight_cost * value;

        clip_gradients(grad, n);
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <decay_type decay, typename V, typename G, cpp_enable_if(decay == decay_type::NONE)>
    void update_grad(const V& value, G& grad, size_t n) {
        clip_gradients(grad, n);

        cpp_unused(value);
    }

    /*!
     * \brief Return the name of the trainer
     */
    static std::string name() {
        return "Stochastic Gradient Descent";
    }
};

} //end of dll namespace
