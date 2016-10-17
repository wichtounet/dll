//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

#include "dll/util/blas.hpp"
#include "dll/util/checks.hpp" // For NaN checks
#include "dll/trainer/sgd_context.hpp" //Context for SGD

namespace dll {

template <typename L, typename Enable = void>
struct extract_function;

template <typename L>
struct extract_function<L, std::enable_if_t<decay_layer_traits<L>::is_standard_layer()>> {
    static constexpr const function activation_function = std::decay_t<L>::activation_function;
};

template <typename L>
struct extract_function<L, std::enable_if_t<decay_layer_traits<L>::is_rbm_layer()>> {
    static_assert(
        std::decay_t<L>::hidden_unit == unit_type::BINARY || std::decay_t<L>::hidden_unit == unit_type::RELU || std::decay_t<L>::hidden_unit == unit_type::SOFTMAX,
        "Only (C)RBM with binary, softmax or RELU hidden unit are supported");

    static constexpr const function activation_function =
        std::decay_t<L>::hidden_unit == unit_type::BINARY
            ? function::SIGMOID
            : (std::decay_t<L>::hidden_unit == unit_type::SOFTMAX ? function::SOFTMAX : function::RELU);
};

template <typename Layer, typename Input, typename Output, typename Errors, cpp_enable_if(decay_layer_traits<Layer>::is_max_pooling_layer() && !decay_layer_traits<Layer>::is_dynamic())>
auto upsample(Input&& input, Output&& output, Errors&& errors) {
    return etl::max_pool_derivative_3d<Layer::C1, Layer::C2, Layer::C3>(input, output) >> etl::upsample_3d<Layer::C1, Layer::C2, Layer::C3>(errors);
}

template <typename Layer, typename Input, typename Output, typename Errors, cpp_enable_if(decay_layer_traits<Layer>::is_max_pooling_layer() && decay_layer_traits<Layer>::is_dynamic())>
auto upsample(Input&& input, Output&& output, Errors&& errors) {
    auto c1 = etl::dim(input, 0) / etl::dim(output, 0);
    auto c2 = etl::dim(input, 1) / etl::dim(output, 1);
    auto c3 = etl::dim(input, 2) / etl::dim(output, 2);

    return etl::max_pool_derivative_3d(input, output, c1, c2, c3) >> etl::upsample_3d(errors, c1, c2, c3);
}

template <typename Layer, typename Input, typename Output, typename Errors, cpp_enable_if(decay_layer_traits<Layer>::is_avg_pooling_layer() && !decay_layer_traits<Layer>::is_dynamic())>
auto upsample(Input&& input, Output&& output, Errors&& errors) {
    return etl::avg_pool_derivative_3d<Layer::C1, Layer::C2, Layer::C3>(input, output) >> etl::upsample_3d<Layer::C1, Layer::C2, Layer::C3>(errors);
}

template <typename Layer, typename Input, typename Output, typename Errors, cpp_enable_if(decay_layer_traits<Layer>::is_avg_pooling_layer() && decay_layer_traits<Layer>::is_dynamic())>
auto upsample(Input&& input, Output&& output, Errors&& errors) {
    auto c1 = etl::dim(input, 0) / etl::dim(output, 0);
    auto c2 = etl::dim(input, 1) / etl::dim(output, 1);
    auto c3 = etl::dim(input, 2) / etl::dim(output, 2);

    return etl::avg_pool_derivative_3d(input, output, c1, c2, c3) >> etl::upsample_3d(errors, c1, c2, c3);
}

template <typename DBN>
struct sgd_trainer {
    using dbn_t     = DBN;
    using weight    = typename dbn_t::weight;
    using this_type = sgd_trainer<dbn_t>;

    static constexpr const auto layers     = dbn_t::layers;
    static constexpr const auto batch_size = dbn_t::batch_size;

    dbn_t& dbn;

    template<std::size_t Layer, typename Enable = void>
    struct input_layer_t {
        static constexpr const std::size_t L = Layer;
    };

    template<std::size_t Layer>
    struct input_layer_t<Layer, std::enable_if_t< decay_layer_traits<typename dbn_t::template layer_type<Layer>>::is_transform_layer() >> {
        static constexpr const std::size_t L = input_layer_t<Layer + 1>::L;
    };

    template<std::size_t L, std::size_t S, typename Layer, cpp_enable_if((L != S && dbn_traits<dbn_t>::is_dynamic()))>
    void back_init(const Layer& input_layer){
        decltype(auto) layer = dbn.template layer_get<L>();

        layer.template get_sgd_context<dbn_t>().output = input_layer.template prepare_input_batch<batch_size>();
        layer.template get_sgd_context<dbn_t>().errors = input_layer.template prepare_input_batch<batch_size>();

        this->template back_init<L+1, S>(input_layer);
    }

    template<std::size_t L, std::size_t S, typename Layer, cpp_enable_if((L != S && !dbn_traits<dbn_t>::is_dynamic()))>
    void back_init(const Layer& input_layer){
        this->template back_init<L+1, S>(input_layer);
    }

    template<std::size_t L, std::size_t S, typename Layer, cpp_enable_if(L == S)>
    void back_init(const Layer& /*input_layer*/){}

    explicit sgd_trainer(dbn_t& dbn) : dbn(dbn) {
        dbn.for_each_layer([](auto& layer) {
            layer.template init_sgd_context<dbn_t>();
        });

        decltype(auto) input_layer = dbn.template layer_get<input_layer_t<0>::L>();
        this->template back_init<0, input_layer_t<0>::L>(input_layer);
    }

    void init_training(std::size_t) {}

    template <typename Sample>
    void compute_outputs(const Sample& item_data) {
        auto& first_layer         = dbn.template layer_get<0>();
        auto& first_layer_context = first_layer.template get_sgd_context<dbn_t>();

        first_layer.batch_activate_hidden(first_layer_context.output, item_data);

        dbn.for_each_layer_pair([](auto& layer_1, auto& layer_2) {
            auto& ctx1 = layer_1.template get_sgd_context<dbn_t>();
            auto& ctx2 = layer_2.template get_sgd_context<dbn_t>();

            layer_2.batch_activate_hidden(ctx2.output, ctx1.output);

            cpp::static_if<decay_layer_traits<decltype(layer_2)>::is_pooling_layer()>([&](auto f) {
                f(ctx2).input = ctx1.output;
            });
        });
    }

    template <typename Layer, typename Weight, typename Grad, typename Inputs, typename Errors,
             cpp_enable_if(decay_layer_traits<Layer>::is_dense_layer() && etl::decay_traits<Inputs>::dimensions() == 2)>
    static void compute_weight_gradients(Layer&, Grad& grad, Inputs& inputs, Errors& errors) {
        dense_compute_weight_gradients<Weight>(grad, inputs, errors);
    }

    template <typename Layer, typename Weight, typename Grad, typename Inputs, typename Errors,
             cpp_enable_if(decay_layer_traits<Layer>::is_dense_layer() && etl::decay_traits<Inputs>::dimensions() != 2)>
    static void compute_weight_gradients(Layer& layer, Grad& grad, Inputs& inputs, Errors& errors) {
        dense_compute_weight_gradients<Weight>(grad, etl::reshape(inputs, batch_size, num_visible(layer)), errors);
    }

#ifndef ETL_BLAS_MODE

    template <typename Weight, typename Grad, typename Inputs, typename Errors>
    static void dense_compute_weight_gradients(Grad& grad, Inputs&& inputs, Errors& errors) {
        for (std::size_t i = 0; i < batch_size; ++i) {
            grad += etl::outer(inputs(i), errors(i));
        }
    }

#else

    template <typename Weight, typename Grad, typename Inputs, typename Errors>
    static void dense_compute_weight_gradients(Grad& grad, Inputs&& inputs, Errors& errors) {
        for (std::size_t i = 0; i < batch_size; ++i) {
            blas_ger(
                etl::dim<1>(inputs), etl::dim<1>(errors),
                1.0,
                inputs(i).memory_start(), errors(i).memory_start(), grad.memory_start());
        }
    }

#endif

    template <typename Layer, typename Weight, typename Grad, typename Inputs, typename Errors,
             cpp_enable_if(decay_layer_traits<Layer>::is_convolutional_layer())>
    static void compute_weight_gradients(Layer& /*layer*/, Grad& grad, Inputs& inputs, Errors& errors) {
        grad = conv_4d_valid_filter_flipped(inputs, errors);
    }

    template <typename Layer, typename Context, typename Inputs,
             cpp_enable_if((decay_layer_traits<Layer>::is_neural_layer()))>
    static void compute_gradients(Layer& layer, Context& ctx, Inputs& inputs) {
        ctx.w_grad = 0;

        compute_weight_gradients<Layer, weight>(layer, ctx.w_grad, inputs, ctx.errors);

        cpp::static_if<decay_layer_traits<Layer>::is_dense_layer()>([&](auto f) {
            f(ctx.b_grad) = etl::sum_l(ctx.errors);
        }).else_([&](auto f) { f(ctx.b_grad) = etl::mean_r(etl::sum_l(f(ctx.errors))); });

        nan_check_deep(ctx.w_grad);
        nan_check_deep(ctx.b_grad);
    }

    template <typename Layer, typename Context, typename Inputs, cpp_enable_if(decay_layer_traits<Layer>::is_pooling_layer() || decay_layer_traits<Layer>::is_transform_layer())>
    static void compute_gradients(Layer&, Context&, Inputs&) {
        //Pooling and transform layers have no weight
    }

    //Backpropagate errors from pooling
    template <typename Layer1, typename Layer2, cpp_enable_if(decay_layer_traits<Layer2>::is_pooling_layer())>
    static void compute_errors(Layer1& r1, Layer2& r2) {
        constexpr const auto a_f = extract_function<Layer1>::activation_function;

        auto& ctx1 = r1.template get_sgd_context<dbn_t>();
        auto& ctx2 = r2.template get_sgd_context<dbn_t>();

        for (std::size_t i = 0; i < batch_size; ++i) {
            ctx1.errors(i) = f_derivative<a_f>(ctx1.output(i)) >> upsample<Layer2>(ctx2.input(i), ctx2.output(i), ctx2.errors(i));
        }

        nan_check_deep(ctx1.errors);
    }

    template <typename Layer1, typename Context1, typename Layer2, typename Context2, typename DF>
    static void compute_errors_from_dense(Layer1&, Context1& ctx1, Layer2& r2, Context2& ctx2, DF derivative) {
        for (std::size_t i = 0; i < batch_size; ++i) {
            ctx1.errors(i) = derivative(i) >> (r2.w * ctx2.errors(i));
        }

        nan_check_deep(ctx1.errors);
    }

    template <typename Layer1, typename Context1, typename Layer2, typename Context2, typename DF>
    static void compute_errors_from_conv(Layer1&, Context1& ctx1, Layer2& r2, Context2& ctx2, DF derivative) {
        ctx1.errors = derivative() >> etl::conv_4d_full_flipped(ctx2.errors, r2.w);

        nan_check_deep(ctx1.errors);
    }

    //Backpropagate errors from dense to (dense or conv)
    template <typename Layer1, typename Layer2,
             cpp_enable_if(decay_layer_traits<Layer1>::is_neural_layer() && decay_layer_traits<Layer2>::is_dense_layer())>
    static void compute_errors(Layer1& r1, Layer2& r2) {
        constexpr const auto a_f = extract_function<Layer1>::activation_function;

        auto& ctx1 = r1.template get_sgd_context<dbn_t>();
        auto& ctx2 = r2.template get_sgd_context<dbn_t>();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
        compute_errors_from_dense(r1, ctx1, r2, ctx2, [&](std::size_t i) { return f_derivative<a_f>(ctx1.output(i)); });
#pragma GCC diagnostic pop
    }

    //Backpropagate errors from conv to (dense or conv)
    template <typename Layer1, typename Layer2,
             cpp_enable_if(decay_layer_traits<Layer1>::is_neural_layer() && decay_layer_traits<Layer2>::is_convolutional_layer())>
    static void compute_errors(Layer1& r1, Layer2& r2) {
        constexpr const auto a_f = extract_function<Layer1>::activation_function;

        auto& ctx1 = r1.template get_sgd_context<dbn_t>();
        auto& ctx2 = r2.template get_sgd_context<dbn_t>();

        compute_errors_from_conv(r1, ctx1, r2, ctx2, [&] { return f_derivative<a_f>(ctx1.output); });
    }

    //Backpropagate errors from dense to pooling
    template <typename Layer1, typename Layer2,
             cpp_enable_if(!decay_layer_traits<Layer1>::is_neural_layer() && decay_layer_traits<Layer2>::is_dense_layer())>
    static void compute_errors(Layer1& r1, Layer2& r2) {
        auto& ctx1 = r1.template get_sgd_context<dbn_t>();
        auto& ctx2 = r2.template get_sgd_context<dbn_t>();

        compute_errors_from_dense(r1, ctx1, r2, ctx2, [](std::size_t) { return 1.0; });
    }

    //Backpropagate errors from conv to pooling
    template <typename Layer1, typename Layer2,
             cpp_enable_if(!decay_layer_traits<Layer1>::is_neural_layer() && decay_layer_traits<Layer2>::is_convolutional_layer())>
    static void compute_errors(Layer1& r1, Layer2& r2) {
        auto& ctx1 = r1.template get_sgd_context<dbn_t>();
        auto& ctx2 = r2.template get_sgd_context<dbn_t>();

        compute_errors_from_conv(r1, ctx1, r2, ctx2, [] { return 1.0; });
    }

    template <typename D, typename It>
    void copy_inputs(D& dest, It first, It last) {
        std::size_t i = 0;

        while (first != last) {
            dest(i++) = *first++;
        }
    }

    template <typename D, typename It>
    void copy_labels(D& dest, It first, It last) {
        std::size_t i = 0;

        while (first != last) {
            for (std::size_t l = 0; l < etl::dim<1>(dest); ++l) {
                dest(i, l) = (*first)[l];
            }
            ++i;
            ++first;
        }
    }

    template <typename Layer, typename Context, typename Labels, cpp_enable_if(decay_layer_traits<Layer>::is_standard_dense_layer())>
    void compute_last_errors(Layer& /*layer*/, Context& context, Labels& labels) {
        constexpr const auto last_a_f = extract_function<Layer>::activation_function;

        context.errors = f_derivative<last_a_f>(context.output) >> (labels - context.output);

        nan_check_deep(context.errors);
    }

    template <typename Layer, typename Context, typename Labels, cpp_enable_if(decay_layer_traits<Layer>::is_dense_rbm_layer())>
    void compute_last_errors(Layer& /*layer*/, Context& context, Labels& labels) {
        constexpr const auto last_a_f = extract_function<Layer>::activation_function;

        static_assert(last_a_f == function::SOFTMAX || last_a_f == function::SIGMOID || last_a_f == function::RELU, "Only softmax/sigmoid/relu RBM can be used as last RBM layer");

        context.errors = f_derivative<last_a_f>(context.output) >> (labels - context.output);

        nan_check_deep(context.errors);
    }

    template <typename T, typename L>
    double train_batch(std::size_t /*epoch*/, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch) {
        cpp_assert(data_batch.size() == label_batch.size(), "Invalid sizes");

        auto n = label_batch.size();

        decltype(auto) input_layer = dbn.template layer_get<input_layer_t<0>::L>();

        decltype(auto) first_layer = dbn.template layer_get<0>();
        decltype(auto) first_ctx = first_layer.template get_sgd_context<dbn_t>();

        decltype(auto) last_layer = dbn.template layer_get<layers - 1>();
        decltype(auto) last_ctx = last_layer.template get_sgd_context<dbn_t>();

        // Prepare initial inputs and final outputs (labels)

        auto inputs = input_layer.template prepare_input_batch<batch_size>();
        auto labels = last_layer.template prepare_output_batch<batch_size>();

        //Copy inputs and labels into suitable data structure

        copy_inputs(inputs, data_batch.begin(), data_batch.end());
        copy_labels(labels, label_batch.begin(), label_batch.end());

        //Feedforward pass

        compute_outputs(inputs);

        static_assert(
            decay_layer_traits<decltype(last_layer)>::is_dense_layer(),
            "The last layer must be dense for SGD trainining");

        //Compute the errors of the last layer

        compute_last_errors(last_layer, last_ctx, labels);

        //Compute the gradients of each layer

        dbn.for_each_layer_rpair([](auto& r1, auto& r2) {
            auto& ctx1 = r1.template get_sgd_context<dbn_t>();
            auto& ctx2 = r2.template get_sgd_context<dbn_t>();

            this_type::compute_gradients(r2, ctx2, ctx1.output);

            this_type::compute_errors(r1, r2);
        });

        compute_gradients(first_layer, first_ctx, inputs);

        //Apply gradients

        dbn.for_each_layer([this, n](auto& layer) {
            this->apply_gradients(layer, n);
        });

        return etl::mean(etl::abs(labels - last_ctx.output));
    }

    template <typename L, cpp_enable_if(decay_layer_traits<L>::is_neural_layer())>
    void apply_gradients(L& layer, std::size_t n) {
        auto& context = layer.template get_sgd_context<dbn_t>();

        //Update the gradients
        this->update_grad(layer.w, context.w_grad, w_decay(dbn_traits<dbn_t>::decay()), 0.0);
        this->update_grad(layer.b, context.b_grad, b_decay(dbn_traits<dbn_t>::decay()), 0.0);

        //Update with momentum and learning rate
        if (dbn_traits<dbn_t>::has_momentum()) {
            auto momentum = dbn.momentum;
            auto eps      = dbn.learning_rate;

            context.w_inc = momentum * context.w_inc + (eps / n) * context.w_grad;
            context.b_inc = momentum * context.b_inc + (eps / n) * context.b_grad;

            layer.w += context.w_inc;
            layer.b += context.b_inc;
        } else {
            auto eps = dbn.learning_rate;

            layer.w += (eps / n) * context.w_grad;
            layer.b += (eps / n) * context.b_grad;
        }

        nan_check_deep(layer.w);
        nan_check_deep(layer.b);
    }

    template <typename L, cpp_enable_if(decay_layer_traits<L>::is_pooling_layer() || decay_layer_traits<L>::is_transform_layer())>
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
            grad = grad - penalty;
        }
    }

    static std::string name() {
        return "Stochastic Gradient Descent";
    }
};

} //end of dll namespace
