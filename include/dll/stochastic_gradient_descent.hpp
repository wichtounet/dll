//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*! \file Stochastic Gradient Descent (SGD) Implementation */

#ifndef DLL_STOCHASTIC_GRADIENT_DESCENT
#define DLL_STOCHASTIC_GRADIENT_DESCENT

#include "context.hpp"

namespace dll {

template<typename RBM>
struct sgd_context {
    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    static constexpr const std::size_t num_visible = rbm_t::num_visible;
    static constexpr const std::size_t num_hidden = rbm_t::num_hidden;

    etl::fast_matrix<weight, num_visible, num_hidden> w_grad;
    etl::fast_vector<weight, num_hidden> b_grad;
    etl::fast_vector<weight, num_visible> c_grad;

    etl::fast_matrix<weight, num_visible, num_hidden> w_inc;
    etl::fast_vector<weight, num_hidden> b_inc;
    etl::fast_vector<weight, num_visible> c_inc;

    etl::fast_vector<weight, num_hidden> o_a;
    etl::fast_vector<weight, num_hidden> o_s;
    etl::fast_vector<weight, num_hidden> errors;

    sgd_context() : w_inc(0), b_inc(0), c_inc(0), o_a(0), o_s(0), errors(0) {}
};

template<typename DBN>
struct sgd_trainer {
    using dbn_t = DBN;
    using weight = typename dbn_t::weight;
    using this_type = sgd_trainer<dbn_t>;

    using rbm_context_tuple_t = typename context_builder<sgd_context, typename DBN::tuple_type>::type;

    static constexpr const std::size_t layers = dbn_t::layers;

    rbm_context_tuple_t rbm_contexts;

    dbn_t& dbn;
    typename dbn_t::tuple_type& tuples;

    sgd_trainer(dbn_t& dbn) : dbn(dbn), tuples(dbn.tuples) {
        cpp::for_each(tuples, [](auto& r1){
            using rbm_t = std::decay_t<decltype(r1)>;

            if(is_relu(rbm_t::hidden_unit)){
                std::cerr << "Warning: SGD is not tuned for RELU units" << std::endl;
            }
        });
    }

    void init_training(std::size_t){}

    template<typename Sample>
    void compute_outputs(const Sample& item_data){
        etl::dyn_vector<typename Sample::value_type> item(item_data);

        auto& first_rbm = dbn.template layer_get<0>();
        auto& first_rbm_context = std::get<0>(rbm_contexts);

        first_rbm.activate_hidden(first_rbm_context.o_a, first_rbm_context.o_s, item, item);

        cpp::for_each_pair(tuples, rbm_contexts, [](auto&, auto& r2, auto& ctx1, auto& ctx2){
            r2.activate_hidden(ctx2.o_a, ctx2.o_s, ctx1.o_a, ctx1.o_s);
        });
    }

    template<typename RBM, typename Context, typename Inputs>
    static void compute_gradients(RBM& , Context& ctx, const Inputs& inputs){
        for(std::size_t i = 0; i < etl::rows(inputs); i++){
            for(std::size_t j = 0; j < etl::rows(ctx.errors); j++){
                ctx.w_grad(i,j) += inputs(i) * ctx.errors(j);
            }
        }

        ctx.b_grad += ctx.errors;

        nan_check_deep_3(ctx.w_grad, ctx.b_grad, ctx.c_grad);
    }

    template<typename T1, typename T2, bool M = dbn_traits<dbn_t>::has_momentum(), cpp::enable_if_u<M> = cpp::detail::dummy>
    static T2& get_fgrad(T1& , T2& inc){
        return inc;
    }

    template<typename T1, typename T2, bool M = dbn_traits<dbn_t>::has_momentum(), cpp::disable_if_u<M> = cpp::detail::dummy>
    static T1& get_fgrad(T1& grad, T2& ){
        return grad;
    }

    template<typename T, typename L>
    void train_batch(std::size_t /*epoch*/, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        cpp_assert(data_batch.size() == label_batch.size(), "Invalid sizes");

        auto n_samples = label_batch.size();

        constexpr const auto n_outputs = dbn_t::template layer_output_size<layers - 1>();

        //Reset the gradients

        cpp::for_each(rbm_contexts, [](auto& context){
            context.w_grad = 0.0;
            context.b_grad = 0.0;
            context.c_grad = 0.0;
        });

        //Compute the total gradients for the mini batch

        auto it = data_batch.begin();
        auto end = data_batch.end();
        auto lit = label_batch.begin();

        while(it != end){
            //Compute the outputs of each layer one after another
            compute_outputs(*it);

            //Compute the errors of the last layer

            auto& last_ctx = std::get<layers - 1>(rbm_contexts);

            for(std::size_t j = 0; j < n_outputs; ++j){
                auto observed = last_ctx.o_a[j];
                auto desired = (*lit)[j];
                last_ctx.errors[j] = observed * (1.0 - observed) * (desired - observed);
            }

            nan_check_deep(last_ctx.errors);

            //Compute the gradients of each layer

            cpp::for_each_rpair_i(tuples, rbm_contexts, [](std::size_t, auto&, auto& r2, auto& ctx1, auto& ctx2){
                this_type::compute_gradients(r2, ctx2, ctx1.o_a);

                ctx1.errors = ctx1.o_a >> (1 - ctx1.o_a) >> (r2.w * ctx2.errors);

                nan_check_deep(ctx1.errors);
            });

            ++it;
            ++lit;
        }

        //Finalize gradients

        cpp::for_each(rbm_contexts, [n_samples](auto& context){
            context.w_grad /= n_samples;
            context.b_grad /= n_samples;
            context.c_grad /= n_samples;

            nan_check_deep_3(context.w_grad, context.b_grad, context.c_grad);
        });

        //Apply gradients

        cpp::for_each(tuples, rbm_contexts, [this](auto& rbm, auto& context){
            //Update the gradients
            this->update_grad(rbm.w, context.w_grad, w_decay(dbn_traits<dbn_t>::decay()), 0.0);
            this->update_grad(rbm.b, context.b_grad, b_decay(dbn_traits<dbn_t>::decay()), 0.0);
            this->update_grad(rbm.c, context.c_grad, b_decay(dbn_traits<dbn_t>::decay()), 0.0);

            //Update with momentum and learning rate
            if(dbn_traits<dbn_t>::has_momentum()){
                auto momentum = dbn.momentum;
                auto eps = dbn.learning_rate;

                context.w_inc = momentum * context.w_inc + eps * context.w_grad;
                context.b_inc = momentum * context.b_inc + eps * context.b_grad;
                context.c_inc = momentum * context.c_inc + eps * context.c_grad;

                nan_check_deep_3(context.w_inc, context.b_inc, context.c_inc);
            } else {
                auto eps = dbn.learning_rate;

                context.w_grad *= eps;
                context.b_grad *= eps;
                context.c_grad *= eps;

                nan_check_deep_3(context.w_grad, context.b_grad, context.c_grad);
            }

            //Apply the final gradients
            rbm.w += this_type::get_fgrad(context.w_grad, context.w_inc);
            rbm.b += this_type::get_fgrad(context.b_grad, context.b_inc);
            rbm.c += this_type::get_fgrad(context.c_grad, context.c_inc);

            nan_check_deep_3(rbm.w, rbm.b, rbm.c);
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
