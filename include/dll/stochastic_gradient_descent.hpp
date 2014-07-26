//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*! \file Stochastic Gradient Descent (SGD) Implementation */

#ifndef DBN_STOCHASTIC_GRADIENT_DESCENT
#define DBN_STOCHASTIC_GRADIENT_DESCENT

namespace dll {

template<typename DBN>
struct sgd_trainer {
    using dbn_t = DBN;
    using weight = typename dbn_t::weight;

    static constexpr const std::size_t layers = dbn_t::layers;

    dbn_t& dbn;
    typename dbn_t::tuple_type& tuples;

    sgd_trainer(dbn_t& dbn) : dbn(dbn), tuples(dbn.tuples) {}

    void init_training(std::size_t /*batch_size*/){
        //TODO 
    }

    template<typename Sample>
    void compute_outputs(const Sample& item_data){
        etl::dyn_vector<typename Sample::value_type> item(item_data);

        auto& first_rbm = dbn.template layer<0>();

        first_rbm.activate_hidden(first_rbm.o_a, first_rbm.o_s, item, item);

        detail::for_each_pair(tuples, [](auto& r1, auto& r2){
            r2.activate_hidden(r2.o_a, r2.o_s, r1.o_a, r1.o_s);
        });
    }

    template<typename RBM, typename Inputs>
    static void compute_gradients(RBM& rbm, const Inputs& inputs){
        using rbm_t = RBM;

        constexpr const auto n_inputs = rbm_t::num_visible;
        constexpr const auto n_outputs = rbm_t::num_hidden;

        //TODO Rewrite that as ETL expressions

        for(std::size_t a = 0; a < n_inputs; ++a){
            for(std::size_t b = 0; b < n_outputs; ++b){
                rbm.w_grad(a,b) += inputs[a] * rbm.errors[b];
            }
        }

        rbm.b_grad += rbm.errors;
    }

    template<typename T, typename L>
    void train_batch(std::size_t /*epoch*/, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        dll_assert(data_batch.size() == label_batch.size(), "Invalid sizes");

        auto n_samples = label_batch.size();

        constexpr const auto n_outputs = dbn_t::template num_hidden<layers - 1>();

        //TODO Update also the lower levels weights

        detail::for_each(tuples, [](auto& rbm){
            rbm.w_grad = 0.0;
            rbm.b_grad = 0.0;
            rbm.c_grad = 0.0;
        });

        for(std::size_t i = 0; i < n_samples; ++i){
            compute_outputs(data_batch[i]);

            auto& last_rbm = dbn.template layer<layers - 1>();

            // Compute dE/dz_j for each output neuron
            for(std::size_t j = 0; j < n_outputs; ++j){
                auto observed = last_rbm.o_a[j];
                auto desired = label_batch[i][j];
                last_rbm.errors[j] = observed * (1 - observed) * (desired - observed);
            }

            detail::for_each_rpair_i(tuples, [](std::size_t, auto& r1, auto& r2){
                compute_gradients(r2, r1.o_a);

                typedef typename std::remove_reference<decltype(r2)>::type r2_t;

                constexpr const auto n_inputs = r2_t::num_visible;
                constexpr const auto n_outputs = r2_t::num_hidden;

                for(std::size_t a = 0; a < n_inputs; ++a){
                    r1.errors[a] = 0.0;
                    for(std::size_t b = 0; b < n_outputs; ++b){
                        r1.errors[a] += r2.errors[b] * r2.w(a,b);
                    }
                    r1.errors[a] *= r1.o_a[a] * (1 - r1.o_a[a]);
                }
            });
        }

        //Finalize gradients

        detail::for_each(tuples, [n_samples](auto& rbm){
            rbm.w_grad /= n_samples;
            rbm.b_grad /= n_samples;
            rbm.c_grad /= n_samples;
        });

        //Apply gradients

        detail::for_each(tuples, [this](auto& rbm){
            auto learning_rate = dbn.learning_rate;

            rbm.w += learning_rate * rbm.w_grad;
            rbm.b += learning_rate * rbm.b_grad;
            rbm.c += learning_rate * rbm.c_grad;
        });
    }
};

} //end of dbn namespace

#endif