//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file Stochastic Gradient Descent (SGD) Implementation for Dense networks
 */

#ifndef DLL_DENSE_STOCHASTIC_GRADIENT_DESCENT
#define DLL_DENSE_STOCHASTIC_GRADIENT_DESCENT

#include "context.hpp"

namespace dll {

template<typename Layer>
struct dense_sgd_context {
    using layer_t = Layer;
    using weight = typename layer_t::weight;

    static constexpr const std::size_t num_visible = layer_t::num_visible;
    static constexpr const std::size_t num_hidden = layer_t::num_hidden;

    etl::fast_matrix<weight, num_visible, num_hidden> w_grad;
    etl::fast_vector<weight, num_hidden> b_grad;

    etl::fast_matrix<weight, num_visible, num_hidden> w_inc;
    etl::fast_vector<weight, num_hidden> b_inc;

    etl::fast_vector<weight, num_hidden> output;
    etl::fast_vector<weight, num_hidden> errors;

    dense_sgd_context() : w_inc(0.0), b_inc(0.0), output(0.0), errors(0.0) {}
};

template<typename DBN>
struct dense_sgd_trainer {
    using dbn_t = DBN;
    using weight = typename dbn_t::weight;
    using this_type = dense_sgd_trainer<dbn_t>;

    using context_tuple_t = typename context_builder<dense_sgd_context, typename DBN::tuple_type>::type;

    static constexpr const std::size_t layers = dbn_t::layers;

    dbn_t& dbn;
    typename dbn_t::tuple_type& tuples;

    context_tuple_t contexts;

    dense_sgd_trainer(dbn_t& dbn) : dbn(dbn), tuples(dbn.tuples) {
        //Nothing else to init
    }

    void init_training(std::size_t){}

    template<typename Sample>
    void compute_outputs(const Sample& item_data){
        etl::dyn_vector<typename Sample::value_type> item(item_data);

        auto& first_layer = dbn.template layer_get<0>();
        auto& first_layer_context = std::get<0>(contexts);

        first_layer.activate_hidden(first_layer_context.output, item);

        cpp::for_each_pair(tuples, contexts, [](auto&, auto& layer_2, auto& ctx1, auto& ctx2){
            layer_2.activate_hidden(ctx2.output, ctx1.output);
        });
    }

    template<typename RBM, typename Context, typename Inputs>
    static void compute_gradients(RBM& , Context& ctx, const Inputs& inputs){
        ctx.w_grad += etl::outer(inputs, ctx.errors);
        ctx.b_grad += ctx.errors;

        nan_check_deep(ctx.w_grad);
        nan_check_deep(ctx.b_grad);
    }

    template<typename T, typename L>
    void train_batch(std::size_t /*epoch*/, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        cpp_assert(data_batch.size() == label_batch.size(), "Invalid sizes");

        auto n = label_batch.size();

        constexpr const auto n_outputs = dbn_t::template layer_output_size<layers - 1>();

        //Reset the gradients

        cpp::for_each(contexts, [](auto& context){
            context.w_grad = 0.0;
            context.b_grad = 0.0;
        });

        //Compute the total gradients for the mini batch

        auto it = data_batch.begin();
        auto end = data_batch.end();
        auto lit = label_batch.begin();

        while(it != end){
            //Compute the outputs of each layer one after another
            compute_outputs(*it);

            //Compute the errors of the last layer

            auto& last_ctx = std::get<layers - 1>(contexts);

            for(std::size_t j = 0; j < n_outputs; ++j){
                auto observed = last_ctx.output[j];
                auto target = (*lit)[j];

                auto derivative = observed * (1.0 - observed); //derivative of the sigmoid function
                last_ctx.errors[j] = derivative * (target - observed);
            }

            nan_check_deep(last_ctx.errors);

            //Compute the gradients of each layer

            cpp::for_each_rpair_i(tuples, contexts, [](std::size_t, auto&, auto& r2, auto& ctx1, auto& ctx2){
                this_type::compute_gradients(r2, ctx2, ctx1.output);

                ctx1.errors = ctx1.output >> (1.0 - ctx1.output) >> (r2.w * ctx2.errors);

                nan_check_deep(ctx1.errors);
            });

            auto& first_layer = std::get<0>(tuples);
            auto& first_ctx   = std::get<0>(contexts);

            compute_gradients(first_layer, first_ctx, *it);

            ++it;
            ++lit;
        }

        //Apply gradients

        cpp::for_each(tuples, contexts, [this, n](auto& rbm, auto& context){
            //Update the gradients
            this->update_grad(rbm.w, context.w_grad, w_decay(dbn_traits<dbn_t>::decay()), 0.0);
            this->update_grad(rbm.b, context.b_grad, b_decay(dbn_traits<dbn_t>::decay()), 0.0);

            //Update with momentum and learning rate
            if(dbn_traits<dbn_t>::has_momentum()){
                auto momentum = dbn.momentum;
                auto eps = dbn.learning_rate;

                context.w_inc = momentum * context.w_inc + (eps / n) * context.w_grad;
                context.b_inc = momentum * context.b_inc + (eps / n) * context.b_grad;

                rbm.w += context.w_inc;
                rbm.b += context.b_inc;
            } else {
                auto eps = dbn.learning_rate;

                rbm.w += (eps / n) * context.w_grad;
                rbm.b += (eps / n) * context.b_grad;
            }

            nan_check_deep(rbm.w);
            nan_check_deep(rbm.b);
        });
    }

    template<typename V, typename G>
    void update_grad(const V& value, G& grad, decay_type decay, double penalty){
        auto weight_cost = dbn.weight_cost;

        if(decay == decay_type::L1){
            grad = grad - weight_cost * abs(value) - penalty;
        } else if(decay == decay_type::L2){
            grad = grad - weight_cost * value - penalty;
        } else {
            grad = grad - penalty;
        }
    }

    static std::string name(){
        return "Stochastic Gradient Descent";
    }
};

} //end of dll namespace

#endif
