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
    static void compute_gradients(RBM& rbm, std::size_t n_samples, const Output& output, const Errors& errors){
        using rbm_t = RBM;

        constexpr const auto n_inputs = rbm_t::num_visible;
        constexpr const auto n_outputs = rbm_t::num_hidden;

        rbm.w_grad = 0.0;
        rbm.b_grad = 0.0;
        rbm.c_grad = 0.0;

        //TODO Rewrite that as ETL expressions

        for(std::size_t i = 0; i < n_samples; ++i){
            for(std::size_t a = 0; a < n_inputs; ++a){
                for(std::size_t b = 0; b < n_outputs; ++b){
                    rbm.w_grad(a,b) += output[i][b] * errors[i][b];
                }
            }

            for(std::size_t b = 0; b < n_outputs; ++b){
                rbm.b_grad(b) += errors[i][b];
            }
        }
    }

    template<typename RBM>
    void apply_gradients(RBM& rbm, std::size_t n_samples){
        using rbm_t = RBM;

        constexpr const auto n_inputs = rbm_t::num_visible;
        constexpr const auto n_outputs = rbm_t::num_hidden;

        auto learning_rate = dbn.learning_rate;

        rbm.w += learning_rate * rbm.w_grad / n_samples;
        rbm.b += learning_rate * rbm.b_grad / n_samples;
        rbm.c += learning_rate * rbm.c_grad / n_samples;
    }

    template<typename T, typename L>
    void train_batch(std::size_t /*epoch*/, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        dll_assert(data_batch.size() == label_batch.size(), "Invalid sizes");

        auto n_samples = label_batch.size();
        constexpr const auto n_outputs = dbn_t::template num_hidden<layers - 1>();

        static std::vector<etl::fast_vector<weight, n_outputs>> outputs;
        outputs.resize(n_samples);

        for(std::size_t i = 0; i < n_samples; ++i){
            dbn.predict_weights(data_batch[i], outputs[i]);
        }

        static std::vector<etl::fast_vector<weight, n_outputs>> errors;
        errors.resize(n_samples);

        // Compute dE/dz_j for each output neuron
        for(std::size_t i = 0; i < n_samples; ++i){
            for(std::size_t j = 0; j < n_outputs; ++j){
                auto output = outputs[i][j];
                errors[i][j] = output * (1 - output) * (label_batch[i][j] - output);
            }
        }

        //TODO Update also the lower levels weights

        compute_gradients(dbn.template layer<layers -1>(), n_samples, outputs, errors);

        apply_gradients(dbn.template layer<layers - 1>(), n_samples);
    }
};

} //end of dbn namespace

#endif