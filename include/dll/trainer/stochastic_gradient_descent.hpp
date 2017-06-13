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

    dbn_t& dbn;                                             ///< The DBN being trained
    decltype(build_context<sgd_context>(dbn)) full_context; ///< The context

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
    explicit sgd_trainer(dbn_t& dbn) : dbn(dbn), full_context(build_context<sgd_context>(dbn)) {
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

        const auto n = etl::dim<0>(inputs);

        auto& first_layer = std::get<0>(full_context).first;
        auto& first_ctx   = *std::get<0>(full_context).second;
        auto& last_ctx    = *std::get<layers - 1>(full_context).second;

        // Ensure that the data batch and the label batch are of the same size
        cpp_assert(etl::dim<0>(inputs) == etl::dim<0>(labels), "Invalid sizes");

        // Ensure that the context can hold the inputs
        cpp_assert(etl::dim<0>(inputs) <= etl::dim<0>(first_ctx.input), "Invalid sizes");

        const bool full_batch = etl::dim<0>(inputs) == etl::dim<0>(first_ctx.input);

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

            first_layer.batch_activate_hidden(first_ctx.output, first_ctx.input);

            cpp::for_each_pair(full_context, [](auto& layer_ctx_1, auto& layer_ctx_2) {
                auto& layer_2 = layer_ctx_2.first;

                auto& ctx1 = *layer_ctx_1.second;
                auto& ctx2 = *layer_ctx_2.second;

                ctx2.input = ctx1.output;
                layer_2.batch_activate_hidden(ctx2.output, ctx2.input);
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

            cpp::for_each(full_context, [this, n](auto& layer_ctx) {
                // Compute the gradients
                layer_ctx.first.compute_gradients(*layer_ctx.second);

                // Apply the gradients
                this->apply_gradients<dbn_traits<dbn_t>::updater()>(layer_ctx.first, *layer_ctx.second, n);
            });
        }

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
    template <updater_type UT, typename L, typename C, cpp_enable_if(decay_layer_traits<L>::is_neural_layer() && UT == updater_type::SGD)>
    void apply_gradients(L& layer, C& context, size_t n) {
        dll::auto_timer timer("sgd::apply_grad");

        //Update the gradients
        this->update_grad<w_decay(dbn_traits<dbn_t>::decay())>(layer.w, context.w_grad, 0.0);
        this->update_grad<b_decay(dbn_traits<dbn_t>::decay())>(layer.b, context.b_grad, 0.0);

        auto eps = dbn.learning_rate;

        layer.w += (eps / n) * context.w_grad;
        layer.b += (eps / n) * context.b_grad;

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(decay_layer_traits<L>::is_neural_layer() && UT == updater_type::MOMENTUM)>
    void apply_gradients(L& layer, C& context, size_t n) {
        dll::auto_timer timer("sgd::apply_grad");

        //Update the gradients
        this->update_grad<w_decay(dbn_traits<dbn_t>::decay())>(layer.w, context.w_grad, 0.0);
        this->update_grad<b_decay(dbn_traits<dbn_t>::decay())>(layer.b, context.b_grad, 0.0);

        //Update with momentum and learning rate
        auto momentum = dbn.momentum;
        auto eps      = dbn.learning_rate;

        // Note(perf): Some performance could be gained by doing the pair of
        // operations on w in a loop to improve data locality

        context.w_inc = momentum * context.w_inc + (eps / n) * context.w_grad;
        layer.w += context.w_inc;

        context.b_inc = momentum * context.b_inc + (eps / n) * context.b_grad;
        layer.b += context.b_inc;

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(decay_layer_traits<L>::is_neural_layer() && UT == updater_type::ADAGRAD)>
    void apply_gradients(L& layer, C& context, size_t n) {
        dll::auto_timer timer("sgd::apply_grad");

        //Update the gradients
        this->update_grad<w_decay(dbn_traits<dbn_t>::decay())>(layer.w, context.w_grad, 0.0);
        this->update_grad<b_decay(dbn_traits<dbn_t>::decay())>(layer.b, context.b_grad, 0.0);

        const auto eps = dbn.learning_rate;
        const auto e = 1e-8;

        context.w_inc = context.w_inc + (context.w_grad >> context.w_grad);
        context.b_inc = context.b_inc + (context.b_grad >> context.b_grad);

        layer.w += (eps >> context.w_grad) / etl::sqrt(context.w_inc + e);
        layer.b += (eps >> context.b_grad) / etl::sqrt(context.b_inc + e);

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(n);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_enable_if(decay_layer_traits<L>::is_neural_layer() && UT == updater_type::RMSPROP)>
    void apply_gradients(L& layer, C& context, size_t n) {
        dll::auto_timer timer("sgd::apply_grad");

        //Update the gradients
        this->update_grad<w_decay(dbn_traits<dbn_t>::decay())>(layer.w, context.w_grad, 0.0);
        this->update_grad<b_decay(dbn_traits<dbn_t>::decay())>(layer.b, context.b_grad, 0.0);

        const auto eps = dbn.learning_rate;
        const auto decay = dbn.rmsprop_decay;
        const auto e = 1e-8;

        context.w_inc = decay * context.w_inc + (1 - decay) * (context.w_grad >> context.w_grad);
        context.b_inc = decay * context.b_inc + (1 - decay) * (context.b_grad >> context.b_grad);

        layer.w += (eps >> context.w_grad) / etl::sqrt(context.w_inc + e);
        layer.b += (eps >> context.b_grad) / etl::sqrt(context.b_inc + e);

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);

        cpp_unused(n);
    }

    /*!
     * \brief Apply the gradients to the given layer
     */
    template <updater_type UT, typename L, typename C, cpp_disable_if(decay_layer_traits<L>::is_neural_layer())>
    void apply_gradients(L& /*layer*/, C& /*context*/, size_t /*n*/) {
        //Pooling and transform layers have no weights, therefore no
        //gradients
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <decay_type decay, typename V, typename G, cpp_enable_if(decay == decay_type::L1)>
    void update_grad(const V& value, G& grad, double penalty) {
        grad = grad - dbn.l1_weight_cost * abs(value) - penalty;
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <decay_type decay, typename V, typename G, cpp_enable_if(decay == decay_type::L2)>
    void update_grad(const V& value, G& grad, double penalty) {
        grad = grad - dbn.l2_weight_cost * value - penalty;
    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <decay_type decay, typename V, typename G, cpp_enable_if(decay == decay_type::L1L2)>
    void update_grad(const V& value, G& grad, double penalty) {
        grad = grad - dbn.l1_weight_cost * abs(value) - dbn.l2_weight_cost * value - penalty;

    }

    /*!
     * \brief Update the given gradients according to the given decay function
     */
    template <decay_type decay, typename V, typename G, cpp_enable_if(decay == decay_type::NONE)>
    void update_grad(const V& value, G& grad, double penalty) {
        if(penalty != 0.0){
            grad = grad - penalty;
        }

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
