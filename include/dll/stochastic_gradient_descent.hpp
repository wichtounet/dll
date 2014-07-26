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

    template<typename RBM, typename Output, typename Errors>
    static void compute_gradients(RBM& rbm, const Output& output, const Errors& errors){
        using rbm_t = RBM;

        constexpr const auto n_inputs = rbm_t::num_visible;
        constexpr const auto n_outputs = rbm_t::num_hidden;

        //TODO Rewrite that as ETL expressions

        for(std::size_t a = 0; a < n_inputs; ++a){
            for(std::size_t b = 0; b < n_outputs; ++b){
                rbm.w_grad(a,b) += output[b] * errors[b];
            }
        }

        rbm.b_grad += errors;
    }

    template<typename RBM>
    void apply_gradients(RBM& rbm){
        using rbm_t = RBM;

        constexpr const auto n_inputs = rbm_t::num_visible;
        constexpr const auto n_outputs = rbm_t::num_hidden;

        auto learning_rate = dbn.learning_rate;

        rbm.w += learning_rate * rbm.w_grad;
        rbm.b += learning_rate * rbm.b_grad;
        rbm.c += learning_rate * rbm.c_grad;
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
            static etl::fast_vector<weight, n_outputs> outputs;

            dbn.predict_weights(data_batch[i], outputs);

            static etl::fast_vector<weight, n_outputs> errors;

            // Compute dE/dz_j for each output neuron
            for(std::size_t j = 0; j < n_outputs; ++j){
                auto observed = outputs[j];
                errors[j] = observed * (1 - observed) * (label_batch[i][j] - observed);
            }

            compute_gradients(dbn.template layer<layers - 1>(), outputs, errors);
        }

        detail::for_each(tuples, [n_samples](auto& rbm){
            rbm.w_grad /= n_samples;
            rbm.b_grad /= n_samples;
            rbm.c_grad /= n_samples;
        });

        apply_gradients(dbn.template layer<layers - 1>());
    }
};

} //end of dbn namespace

#endif