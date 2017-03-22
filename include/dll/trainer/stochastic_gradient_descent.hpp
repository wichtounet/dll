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

#include "dll/trainer/sgd_context.hpp" //Context for SGD
#include "dll/util/batch.hpp"          // Create batches
#include "dll/util/checks.hpp"         // For NaN checks
#include "dll/util/timers.hpp"         // For auto_timer

namespace dll {

template <typename DBN>
struct sgd_trainer {
    using dbn_t     = DBN;
    using weight    = typename dbn_t::weight;
    using this_type = sgd_trainer<dbn_t>;

    static constexpr const auto layers     = dbn_t::layers;
    static constexpr const auto batch_size = dbn_t::batch_size;

    bool ae_training = false;

    dbn_t& dbn;

    /*!
     * \brief Indicates if the model is being trained as an auto-encoder (true) or not (false)
     */
    void set_autoencoder(bool ae){
        this->ae_training = ae;;
    }

    template<std::size_t Layer, typename Enable = void>
    struct input_layer_t {
        static constexpr const std::size_t L = Layer;
    };

    template<std::size_t Layer>
    struct input_layer_t<Layer, std::enable_if_t< decay_layer_traits<typename dbn_t::template layer_type<Layer>>::is_transform_layer() >> {
        static constexpr const std::size_t L = input_layer_t<Layer + 1>::L;
    };

    // Some Transform layers need to inherit dimensions from back

    template<typename L1, typename L2, cpp_enable_if(decay_layer_traits<L1>::is_transform_layer())>
    static void inherit_from_back(L1& l1, L2& l2){
        auto& ctx1 = l1.template get_sgd_context<dbn_t>();
        auto& ctx2 = l2.template get_sgd_context<dbn_t>();

        if (ctx1.errors.size() == 0) {
            ctx1.output = ctx2.input;
            ctx1.errors = ctx2.input;
            ctx1.input  = ctx2.input;
        }
    }

    template<typename L1, typename L2, cpp_disable_if(decay_layer_traits<L1>::is_transform_layer())>
    static void inherit_from_back(L1& /*l1*/, L2& /*l2*/){ }

    // Some Transform layers need to inherit dimensions from back

    template<typename L1, typename L2, cpp_enable_if(decay_layer_traits<L2>::is_transform_layer())>
    static void inherit_from_front(L1& l1, L2& l2){
        auto& ctx1 = l1.template get_sgd_context<dbn_t>();
        auto& ctx2 = l2.template get_sgd_context<dbn_t>();

        if (ctx2.errors.size() == 0) {
            ctx2.output = ctx1.output;
            ctx2.errors = ctx1.output;
            ctx2.input  = ctx1.output;
        }
    }

    template<typename L1, typename L2, cpp_disable_if(decay_layer_traits<L2>::is_transform_layer())>
    static void inherit_from_front(L1& /*l1*/, L2& /*l2*/){ }

    explicit sgd_trainer(dbn_t& dbn) : dbn(dbn) {
        // Initialize all the SGD contexts
        dbn.for_each_layer([](auto& layer) {
            layer.template init_sgd_context<dbn_t>();
        });

        // Inherit dimensions from back

        dbn.for_each_layer_rpair([](auto& l1, auto& l2) {
            constexpr bool l1_transform = decay_layer_traits<decltype(l1)>::is_transform_layer();
            constexpr bool l2_transform = decay_layer_traits<decltype(l2)>::is_transform_layer();

            if (l1_transform && (!l2_transform || l2.template get_sgd_context<dbn_t>().errors.size())) {
                this_type::inherit_from_back(l1, l2);
            }
        });

        // Inherit dimensions from front

        dbn.for_each_layer_pair([](auto& l1, auto& l2) {
            constexpr bool l2_transform = decay_layer_traits<decltype(l2)>::is_transform_layer();

            if (l2_transform) {
                this_type::inherit_from_front(l1, l2);
            }
        });
    }

    void init_training(std::size_t) {}

    template <typename D, typename It>
    void copy_inputs(D& dest, It first, It last) {
        std::size_t i = 0;

        while (first != last) {
            dest(i++) = *first++;
        }
    }

    template <typename D, typename It, cpp_enable_if(etl::decay_traits<D>::dimensions() == 2)>
    void copy_labels(D& dest, It first, It last) {
        //TODO How does that work in auto encoder mode ?

        std::size_t i = 0;

        while (first != last) {
            for (std::size_t l = 0; l < etl::dim<1>(dest); ++l) {
                dest(i, l) = (*first)[l];
            }
            ++i;
            ++first;
        }
    }

    template <typename D, typename It, cpp_enable_if(etl::decay_traits<D>::dimensions() == 4)>
    void copy_labels(D& dest, It first, It last) {
        //TODO How does that work in auto encoder mode ?

        std::size_t i = 0;

        while (first != last) {
            dest(i++) = *first++;
        }
    }

    // TODO: There are way too many copies going in this function

    template <typename Inputs, typename Labels, typename InputTransformer>
    std::pair<double, double> train_batch(std::size_t /*epoch*/, const Inputs& inputs, const Labels& labels, InputTransformer input_transformer) {
        dll::auto_timer timer("sgd::train_batch");

        // Ensure that the data batch and the label batch are of the same size
        cpp_assert(etl::dim<0>(inputs) == etl::dim<0>(labels), "Invalid sizes");

        const auto n = etl::dim<0>(inputs);

        decltype(auto) first_layer = dbn.template layer_get<0>();
        decltype(auto) first_ctx = first_layer.template get_sgd_context<dbn_t>();

        decltype(auto) last_layer = dbn.template layer_get<layers - 1>();
        decltype(auto) last_ctx = last_layer.template get_sgd_context<dbn_t>();

        const bool full_batch = etl::dim<0>(inputs) == etl::dim<0>(first_ctx.input);

        //Copy inputs into suitable data structure

        auto tilde_inputs = inputs;
        for(size_t i = 0; i < etl::dim<0>(tilde_inputs); ++i){
            input_transformer(tilde_inputs(i));
        }

        //Feedforward pass

        {
            dll::auto_timer timer("sgd::forward");

            if(cpp_unlikely(!full_batch)){
                first_ctx.input  = 0;
                first_ctx.output = 0;

                for (size_t i = 0; i < etl::dim<0>(inputs); ++i) {
                    first_ctx.input(i) = tilde_inputs(i);
                }
            } else {
                first_ctx.input = tilde_inputs;
            }

            first_layer.batch_activate_hidden(first_ctx.output, first_ctx.input);

            dbn.for_each_layer_pair([](auto& layer_1, auto& layer_2) {
                auto& ctx1 = layer_1.template get_sgd_context<dbn_t>();
                auto& ctx2 = layer_2.template get_sgd_context<dbn_t>();

                ctx2.input = ctx1.output;
                layer_2.batch_activate_hidden(ctx2.output, ctx2.input);
            });
        }

        //Compute the errors of the last layer

        if (cpp_unlikely(!full_batch)) {
            first_ctx.input = 0;
            last_ctx.errors = 0;

            for (size_t i = 0; i < etl::dim<0>(inputs); ++i) {
                last_ctx.errors(i) = labels(i) - last_ctx.output(i);
            }
        } else {
            last_ctx.errors = labels - last_ctx.output;
        }

        // Backpropagate the error

        {
            dll::auto_timer timer("sgd::backward");

            dbn.for_each_layer_rpair([](auto& r1, auto& r2) {
                auto& ctx1 = r1.template get_sgd_context<dbn_t>();
                auto& ctx2 = r2.template get_sgd_context<dbn_t>();

                r2.adapt_errors(ctx2);
                r2.backward_batch(ctx1.errors, ctx2);
            });

            first_layer.adapt_errors(first_ctx);
        }

        // Compute and apply the gradients

        {
            dll::auto_timer timer("sgd::grad");

            dbn.for_each_layer([this, n](auto& layer) {
                // Compute the gradients
                layer.compute_gradients(layer.template get_sgd_context<dbn_t>());

                // Apply the gradients
                this->apply_gradients(layer, n);
            });
        }

        // Compute error and loss

        double error = 0.0;
        double loss = 0.0;

        {
            dll::auto_timer timer("sgd::error");

            auto& out = last_ctx.output;

            if (cpp_unlikely(!full_batch)) {
                error = etl::mean(etl::abs(labels - slice(out, 0, etl::dim<0>(inputs))));

                if (ae_training) {
                    // Reconstruction Cross-Entropy Loss
                    loss = -sum((labels >> log(slice(out, 0, etl::dim<0>(inputs)))) + ((1.0 - labels) >> log(1 - slice(out, 0, etl::dim<0>(inputs))))) / double(n);
                } else {
                    // Cross-Entropy Loss
                    loss = -sum(log(slice(out, 0, etl::dim<0>(inputs))) >> labels) / double(n);
                }
            } else {
                error = etl::mean(etl::abs(labels - out));

                if (ae_training) {
                    // Reconstruction Cross-Entropy Loss
                    loss = -sum((labels >> log(out)) + ((1.0 - labels) >> log(1 - out))) / double(n);
                } else {
                    // Cross-Entropy Loss
                    loss = -sum(log(out) >> labels) / double(n);
                }
            }
        }

        return std::make_pair(error, loss);
    }

    template <typename L, cpp_enable_if(decay_layer_traits<L>::is_neural_layer())>
    void apply_gradients(L& layer, std::size_t n) {
        dll::auto_timer timer("sgd::apply_grad");

        auto& context = layer.template get_sgd_context<dbn_t>();

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

    template <typename L, cpp_disable_if(decay_layer_traits<L>::is_neural_layer())>
    void apply_gradients(L&, std::size_t) {
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
