//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
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
};

template<typename DBN>
struct sgd_trainer {
    using dbn_t = DBN;
    using weight = typename dbn_t::weight;

    using rbm_context_tuple_t = typename context_builder<sgd_context, typename DBN::tuple_type>::type;

    static constexpr const std::size_t layers = dbn_t::layers;

    rbm_context_tuple_t rbm_contexts;

    dbn_t& dbn;
    typename dbn_t::tuple_type& tuples;

    sgd_trainer(dbn_t& dbn) : dbn(dbn), tuples(dbn.tuples) {}

    void init_training(std::size_t){}

    template<typename Sample>
    void compute_outputs(const Sample& item_data){
        etl::dyn_vector<typename Sample::value_type> item(item_data);

        auto& first_rbm = dbn.template layer<0>();
        auto& first_rbm_context = std::get<0>(rbm_contexts);

        first_rbm.activate_hidden(first_rbm_context.o_a, first_rbm_context.o_s, item, item);

        detail::for_each_pair(tuples, rbm_contexts, [](auto&, auto& r2, auto& ctx1, auto& ctx2){
            r2.activate_hidden(ctx2.o_a, ctx2.o_s, ctx1.o_a, ctx1.o_s);
        });
    }

    template<typename RBM, typename Context, typename Inputs>
    static void compute_gradients(RBM& , Context& ctx, const Inputs& inputs){
        using namespace etl;

        using rbm_t = RBM;

        static fast_matrix<weight, rbm_t::num_visible, rbm_t::num_hidden> t;

        ctx.w_grad += etl::mmul(reshape<rbm_t::num_visible, 1>(inputs), reshape<1, rbm_t::num_hidden>(ctx.errors), t);

        ctx.b_grad += ctx.errors;
    }

    template<typename T1, typename T2, bool M = dbn_traits<dbn_t>::has_momentum(), cpp::enable_if_u<M> = cpp::detail::dummy>
    T2& get_fgrad(T1& , T2& inc){
        return inc;
    }

    template<typename T1, typename T2, bool M = dbn_traits<dbn_t>::has_momentum(), cpp::disable_if_u<M> = cpp::detail::dummy>
    T1& get_fgrad(T1& grad, T2& ){
        return grad;
    }

    template<typename T, typename L>
    void train_batch(std::size_t /*epoch*/, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        cpp_assert(data_batch.size() == label_batch.size(), "Invalid sizes");

        auto n_samples = label_batch.size();

        constexpr const auto n_outputs = dbn_t::template num_hidden<layers - 1>();

        detail::for_each(rbm_contexts, [](auto& context){
            context.w_grad = 0.0;
            context.b_grad = 0.0;
            context.c_grad = 0.0;

            if(dbn_traits<dbn_t>::has_momentum()){
                context.w_inc = 0.0;
                context.b_inc = 0.0;
                context.c_inc = 0.0;
            }
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
                last_ctx.errors[j] = observed * (1 - observed) * (desired - observed);
            }

            //Compute the gradients of each layer

            detail::for_each_rpair_i(tuples, rbm_contexts, [](std::size_t, auto&, auto& r2, auto& ctx1, auto& ctx2){
                compute_gradients(r2, ctx2, ctx1.o_a);

                typedef typename std::remove_reference<decltype(r2)>::type r2_t;

                using namespace etl;

                static fast_matrix<weight, r2_t::num_visible, 1> t;

                ctx1.errors = ctx1.o_a * (1 - ctx1.o_a) * mmul(r2.w, reshape<n_outputs, 1>(ctx2.errors), t);
            });

            ++it;
            ++lit;
        }

        //Finalize gradients

        detail::for_each(rbm_contexts, [n_samples](auto& context){
            context.w_grad /= n_samples;
            context.b_grad /= n_samples;
            context.c_grad /= n_samples;
        });

        //Apply gradients

        detail::for_each(tuples, rbm_contexts, [this](auto& rbm, auto& context){
            //Update momentum gradients
            if(dbn_traits<dbn_t>::has_momentum()){
                auto momentum = dbn.momentum;

                context.w_inc = momentum * context.w_inc + (1 - momentum) * context.w_grad;
                context.b_inc = momentum * context.b_inc + (1 - momentum) * context.b_grad;
                context.c_inc = momentum * context.c_inc + (1 - momentum) * context.c_grad;
            }

            //The final gradients;
            const auto& w_fgrad = get_fgrad(context.w_grad, context.w_inc);
            const auto& b_fgrad = get_fgrad(context.b_grad, context.b_inc);
            const auto& c_fgrad = get_fgrad(context.c_grad, context.c_inc);

            update(rbm.w, w_fgrad, w_decay(dbn_traits<dbn_t>::decay()), 0.0);
            update(rbm.b, b_fgrad, b_decay(dbn_traits<dbn_t>::decay()), 0.0);
            update(rbm.c, c_fgrad, b_decay(dbn_traits<dbn_t>::decay()), 0.0);
        });
    }

    template<typename V, typename G>
    void update(V& value, const G& grad, decay_type decay, double penalty){
        auto learning_rate = dbn.learning_rate;
        auto weight_cost = dbn.weight_cost;

        if(decay == decay_type::L1){
            value += learning_rate * grad - learning_rate * weight_cost * abs(value) - penalty;
        } else if(decay == decay_type::L2){
            value += learning_rate * grad - learning_rate * weight_cost * value - penalty;
        } else {
            value += learning_rate * grad - penalty;
        }
    }

    static std::string name(){
        return "Stochastic Gradient Descent";
    }
};

} //end of dbn namespace

#endif