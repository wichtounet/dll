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

#include "dll/util/checks.hpp"         // For NaN checks
#include "dll/util/timers.hpp"         // For auto_timer

namespace dll {

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

template<template<typename, typename, size_t> class Context, typename DBN>
auto build_context(DBN& dbn){
    return build_context<Context>(dbn, std::make_index_sequence<DBN::layers>());
}

template <typename DBN>
struct sgd_trainer {
    using dbn_t     = DBN;
    using weight    = typename dbn_t::weight;
    using this_type = sgd_trainer<dbn_t>;

    static constexpr auto layers     = dbn_t::layers;
    static constexpr auto batch_size = dbn_t::batch_size;

    dbn_t& dbn;
    decltype(build_context<sgd_context>(dbn)) full_context;

    // Transform layers need to inherit dimensions from back

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

    template<typename L1, typename L2, cpp_disable_if(decay_layer_traits<typename L2::first_type>::is_transform_layer())>
    static void inherit_from_front(L1& /*l1*/, L2& /*l2*/){ }

    explicit sgd_trainer(dbn_t& dbn) : dbn(dbn), full_context(build_context<sgd_context>(dbn)) {
        // Inherit dimensions from front to end (for transform layers)

        cpp::for_each_pair(full_context, [](auto& layer_ctx_1, auto& layer_ctx_2) {
            constexpr bool l2_transform = decay_layer_traits<decltype(layer_ctx_2.first)>::is_transform_layer();

            if (l2_transform) {
                this_type::inherit_from_front(layer_ctx_1, layer_ctx_2);
            }
        });
    }

    void init_training(size_t) {}

    template <typename D, typename It>
    void copy_inputs(D& dest, It first, It last) {
        size_t i = 0;

        while (first != last) {
            dest(i++) = *first++;
        }
    }

    template <typename D, typename It, cpp_enable_if(etl::decay_traits<D>::dimensions() == 2)>
    void copy_labels(D& dest, It first, It last) {
        //TODO How does that work in auto encoder mode ?

        size_t i = 0;

        while (first != last) {
            for (size_t l = 0; l < etl::dim<1>(dest); ++l) {
                dest(i, l) = (*first)[l];
            }
            ++i;
            ++first;
        }
    }

    template <typename D, typename It, cpp_enable_if(etl::decay_traits<D>::dimensions() == 4)>
    void copy_labels(D& dest, It first, It last) {
        //TODO How does that work in auto encoder mode ?

        size_t i = 0;

        while (first != last) {
            dest(i++) = *first++;
        }
    }

    // TODO: There are way too many copies going in this function

    template <typename Inputs, typename Labels, typename InputTransformer>
    std::pair<double, double> train_batch(size_t epoch, const Inputs& inputs, const Labels& labels, InputTransformer input_transformer) {
        //Copy inputs into suitable data structure

        auto tilde_inputs = force_temporary(inputs);
        for(size_t i = 0; i < etl::dim<0>(tilde_inputs); ++i){
            input_transformer(tilde_inputs(i));
        }

        return train_batch(epoch, tilde_inputs, labels);
    }

    template <typename Inputs, typename Labels>
    std::pair<double, double> train_batch(size_t /*epoch*/, const Inputs& inputs, const Labels& labels) {
        dll::auto_timer timer("sgd::train_batch");

        // Ensure that the data batch and the label batch are of the same size
        cpp_assert(etl::dim<0>(inputs) == etl::dim<0>(labels), "Invalid sizes");

        const auto n = etl::dim<0>(inputs);

        auto& first_layer = std::get<0>(full_context).first;
        auto& first_ctx   = *std::get<0>(full_context).second;

        auto& last_ctx   = *std::get<layers - 1>(full_context).second;

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

            // Note: This only compute the derivative of the error function
            // The derivative of the activation function for the last layer
            // will be computed in the next step in adapt errors

            // TODO: The computation is not totally since it should in
            // fact depends on the activation function of the last layer

            // TODO: The MSE and CCE gradients should not be the same

            if /*constexpr*/ (dbn_t::loss == loss_function::CATEGORICAL_CROSS_ENTROPY){
                if (cpp_unlikely(!full_batch)) {
                    last_ctx.errors = 0;

                    for (size_t i = 0; i < n; ++i) {
                        last_ctx.errors(i) = labels(i) - last_ctx.output(i);
                    }
                } else {
                    last_ctx.errors = labels - last_ctx.output;
                }
            } else if (dbn_t::loss == loss_function::MEAN_SQUARED_ERROR){
                if (cpp_unlikely(!full_batch)) {
                    last_ctx.errors = 0;

                    for (size_t i = 0; i < n; ++i) {
                        last_ctx.errors(i) = labels(i) - last_ctx.output(i);
                    }
                } else {
                    last_ctx.errors = labels - last_ctx.output;
                }
            } else if (dbn_t::loss == loss_function::BINARY_CROSS_ENTROPY){
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
            } else {
                cpp_unreachable("Unsupported loss function");
            }

            // Backpropagate the error

            cpp::for_each_rpair(full_context, [](auto& layer_ctx_1, auto& layer_ctx_2) {
                auto& r2 = layer_ctx_2.first;

                auto& ctx1 = *layer_ctx_1.second;
                auto& ctx2 = *layer_ctx_2.second;

                r2.adapt_errors(ctx2);
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
                this->apply_gradients(layer_ctx.first, *layer_ctx.second, n);
            });
        }

        // Compute error and loss

        double error = 0.0;
        double loss = 0.0;

        {
            dll::auto_timer timer("sgd::error");

            auto& out = last_ctx.output;

            if /*constexpr*/ (dbn_t::loss == loss_function::CATEGORICAL_CROSS_ENTROPY){
                if (cpp_unlikely(!full_batch)) {
                    auto sout = slice(out, 0, n);

                    loss = (-1.0 / n) * sum(log(sout) >> labels);
                } else {
                    loss = (-1.0 / n) * sum(log(out) >> labels);
                }
            } else if (dbn_t::loss == loss_function::MEAN_SQUARED_ERROR){
                if (cpp_unlikely(!full_batch)) {
                    auto sout = slice(out, 0, n);

                    loss = (1.0 / 2.0 * n) * sum((sout - labels) >> (sout - labels));
                } else {
                    loss = (1.0 / 2.0 * n) * sum((out - labels) >> (out - labels));
                }
            } else if (dbn_t::loss == loss_function::BINARY_CROSS_ENTROPY){
                if (cpp_unlikely(!full_batch)) {
                    auto sout = slice(out, 0, n);

                    loss = -mean((labels >> log(sout)) + ((1.0 - labels) >> log(1.0 - sout)));
                } else {
                    loss = -mean((labels >> log(out)) + ((1.0 - labels) >> log(1.0 - out)));
                }
            } else {
                cpp_unreachable("Unsupported loss function");
            }

            // Compute the error

            if (cpp_unlikely(!full_batch)) {
                error = amean(labels - slice(out, 0, n));
            } else {
                error = amean(labels - out);
            }
        }

        return std::make_pair(error, loss);
    }

    template <typename L, typename C, cpp_enable_if(decay_layer_traits<L>::is_neural_layer())>
    void apply_gradients(L& layer, C& context, size_t n) {
        dll::auto_timer timer("sgd::apply_grad");

        //Update the gradients
        this->update_grad(layer.w, context.w_grad, w_decay(dbn_traits<dbn_t>::decay()), 0.0);
        this->update_grad(layer.b, context.b_grad, b_decay(dbn_traits<dbn_t>::decay()), 0.0);

        //Update with momentum and learning rate
        if (dbn_traits<dbn_t>::has_momentum()) {
            auto momentum = dbn.momentum;
            auto eps      = dbn.learning_rate;

            // Note(perf): Some performance could be gained by doing the pair of
            // operations on w in a loop to improve data locality

            context.w_inc = momentum * context.w_inc + (eps / n) * context.w_grad;
            layer.w += context.w_inc;

            context.b_inc = momentum * context.b_inc + (eps / n) * context.b_grad;
            layer.b += context.b_inc;
        } else {
            auto eps = dbn.learning_rate;

            layer.w += (eps / n) * context.w_grad;
            layer.b += (eps / n) * context.b_grad;
        }

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);
    }

    template <typename L, typename C, cpp_disable_if(decay_layer_traits<L>::is_neural_layer())>
    void apply_gradients(L& /*layer*/, C& /*context*/, size_t /*n*/) {
        //Pooling and transform layers have no weights, therefore no
        //gradients
    }

    template <typename V, typename G>
    void update_grad(const V& value, G& grad, decay_type decay, double penalty) {
        if (decay == decay_type::L1) {
            grad = grad - dbn.l1_weight_cost * abs(value) - penalty;
        } else if (decay == decay_type::L2) {
            grad = grad - dbn.l2_weight_cost * value - penalty;
        } else if (decay == decay_type::L1L2) {
            grad = grad - dbn.l1_weight_cost * abs(value) - dbn.l2_weight_cost * value - penalty;
        } else {
            if(penalty != 0.0){
                grad = grad - penalty;
            }
        }
    }

    static std::string name() {
        return "Stochastic Gradient Descent";
    }
};

} //end of dll namespace
