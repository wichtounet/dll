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

template<typename DBN, typename Layer>
struct dense_sgd_context {
    using layer_t = Layer;
    using dbn_t = DBN;
    using weight = typename layer_t::weight;

    static constexpr const auto num_visible = layer_t::num_visible;
    static constexpr const auto num_hidden = layer_t::num_hidden;
    static constexpr const auto batch_size = dbn_t::batch_size;

    etl::fast_matrix<weight, num_visible, num_hidden> w_inc;
    etl::fast_matrix<weight, num_hidden> b_inc;

    etl::fast_matrix<weight, batch_size, num_hidden> output;
    etl::fast_matrix<weight, batch_size, num_hidden> errors;

    etl::fast_matrix<weight, num_visible, num_hidden> w_grad;
    etl::fast_matrix<weight, num_hidden> b_grad;

    dense_sgd_context() : w_inc(0.0), b_inc(0.0), output(0.0), errors(0.0) {}
};

template<typename DBN>
struct dense_sgd_trainer {
    using dbn_t = DBN;
    using weight = typename dbn_t::weight;
    using this_type = dense_sgd_trainer<dbn_t>;

    using context_tuple_t = typename dbn_context_builder<dense_sgd_context, dbn_t, typename dbn_t::tuple_type>::type;

    static constexpr const auto layers = dbn_t::layers;
    static constexpr const auto batch_size = dbn_t::batch_size;

    dbn_t& dbn;
    typename dbn_t::tuple_type& tuples;

    context_tuple_t contexts;

    dense_sgd_trainer(dbn_t& dbn) : dbn(dbn), tuples(dbn.tuples) {
        //Nothing else to init
    }

    void init_training(std::size_t){}

    template<typename Sample>
    void compute_outputs(const Sample& item_data){
        auto& first_layer = dbn.template layer_get<0>();
        auto& first_layer_context = std::get<0>(contexts);

        first_layer.batch_activate_hidden(first_layer_context.output, item_data);

        cpp::for_each_pair(tuples, contexts, [](auto&, auto& layer_2, auto& ctx1, auto& ctx2){
            layer_2.batch_activate_hidden(ctx2.output, ctx1.output);
        });
    }

#ifndef ETL_BLAS_MODE

    template<typename Weight, typename Grad, typename Inputs, typename Errors>
    static void compute_weight_gradients(Grad& grad, Inputs& inputs, Errors& errors){
        for(std::size_t i = 0; i < batch_size; ++i){
            grad += etl::outer(inputs(i), errors(i));
        }
    }

#else

    template<typename Weight, typename Grad, typename Inputs, typename Errors, cpp_enable_if(std::is_same<Weight, float>::value)>
    static void compute_weight_gradients(Grad& grad, Inputs& inputs, Errors& errors){
        for(std::size_t i = 0; i < batch_size; ++i){
            cblas_sger(
                CblasRowMajor,
                etl::dim<1>(inputs), etl::dim<1>(errors),
                1.0f,
                inputs(i).memory_start(), 1,
                errors(i).memory_start(), 1,
                grad.memory_start(), etl::dim<1>(errors)
            );
        }
    }

    template<typename Weight, typename Grad, typename Inputs, typename Errors, cpp_enable_if(std::is_same<Weight, double>::value)>
    static void compute_weight_gradients(Grad& grad, Inputs& inputs, Errors& errors){
        for(std::size_t i = 0; i < batch_size; ++i){
            cblas_dger(
                CblasRowMajor,
                etl::dim<1>(inputs), etl::dim<1>(errors),
                1.0,
                inputs(i).memory_start(), 1,
                errors(i).memory_start(), 1,
                grad.memory_start(), etl::dim<1>(errors)
            );
        }
    }

#endif

    template<typename RBM, typename Context, typename Inputs>
    static void compute_gradients(RBM& , Context& ctx, const Inputs& inputs){
        ctx.w_grad = 0;

        compute_weight_gradients<weight>(ctx.w_grad, inputs, ctx.errors);

        ctx.b_grad = etl::sum_l(ctx.errors);

        nan_check_deep(ctx.w_grad);
        nan_check_deep(ctx.b_grad);
    }

    template<typename D, typename It>
    void copy_inputs(D& dest, It first, It last){
        std::size_t i = 0;

        while(first != last){
            dest(i++) = *first++;
        }
    }

    template<typename D, typename It>
    void copy_labels(D& dest, It first, It last){
        std::size_t i = 0;

        while(first != last){
            for(std::size_t l = 0; l < etl::dim<1>(dest); ++l){
                dest(i, l) = (*first)[l];
            }
            ++i;
            ++first;
        }
    }

    template<typename T, typename L>
    void train_batch(std::size_t /*epoch*/, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        cpp_assert(data_batch.size() == label_batch.size(), "Invalid sizes");

        auto n = label_batch.size();

        constexpr const auto n_inputs = dbn_t::template layer_input_size<0>();
        constexpr const auto n_outputs = dbn_t::template layer_output_size<layers - 1>();

        etl::fast_dyn_matrix<weight, batch_size, n_inputs> inputs;
        etl::fast_dyn_matrix<weight, batch_size, n_outputs> labels;

        //Copy inputs and labels into suitable data structure

        copy_inputs(inputs, data_batch.begin(), data_batch.end());
        copy_labels(labels, label_batch.begin(), label_batch.end());

        //Feedforward pass

        compute_outputs(inputs);

        auto& first_layer = std::get<0>(tuples);
        auto& first_ctx   = std::get<0>(contexts);
        auto& last_ctx = std::get<layers - 1>(contexts);

        //Compute the errors of the last layer

        constexpr const auto last_a_f = dbn_t::template layer_type<layers - 1>::activation_function;

        for(std::size_t j = 0; j < n_outputs; ++j){
            auto& observed = last_ctx.output;

            if(last_a_f == function::IDENTITY){
                last_ctx.errors = 1.0                            >> (labels - observed);
            } else if(last_a_f == function::SIGMOID){
                last_ctx.errors = observed >> (1.0 - observed)   >> (labels - observed);
            } else if(last_a_f == function::TANH){
                last_ctx.errors = (1.0 - (observed >> observed)) >> (labels - observed);
            }
        }

        nan_check_deep(last_ctx.errors);

        //Compute the gradients of each layer

            cpp::for_each_rpair_i(tuples, contexts, [](std::size_t, auto& r1, auto& r2, auto& ctx1, auto& ctx2){
                this_type::compute_gradients(r2, ctx2, ctx1.output);

                constexpr const auto a_f = std::decay_t<decltype(r1)>::activation_function;

                for(std::size_t i = 0; i < batch_size; ++i){
                    if(a_f == function::IDENTITY){
                        ctx1.errors(i) = 1.0                                        >> (r2.w * ctx2.errors(i));
                    } else if(a_f == function::SIGMOID){
                        ctx1.errors(i) = ctx1.output(i) >> (1.0 - ctx1.output(i))   >> (r2.w * ctx2.errors(i));
                    } else if(a_f == function::TANH){
                        ctx1.errors(i) = (1.0 - (ctx1.output(i) >> ctx1.output(i))) >> (r2.w * ctx2.errors(i));
                    }
                }

                nan_check_deep(ctx1.errors);
            });

            compute_gradients(first_layer, first_ctx, inputs);

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
        if(decay == decay_type::L1){
            grad = grad - dbn.l1_weight_cost * abs(value) - penalty;
        } else if(decay == decay_type::L2){
            grad = grad - dbn.l2_weight_cost * value - penalty;
        } else if(decay == decay_type::L1L2){
            grad = grad - dbn.l1_weight_cost * abs(value) - dbn.l2_weight_cost * value - penalty;
        } else {
            grad = grad - penalty;
        }
    }

    static std::string name(){
        return "Stochastic Gradient Descent (Dense)";
    }
};

} //end of dll namespace

#endif
