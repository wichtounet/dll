//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*! \file Stochastic Gradient Descent (SGD) Implementation */

#ifndef DLL_STOCHASTIC_GRADIENT_DESCENT
#define DLL_STOCHASTIC_GRADIENT_DESCENT

namespace dll {

template<typename DBN>
struct sgd_trainer {
    using dbn_t = DBN;
    using weight = typename dbn_t::weight;

    static constexpr const std::size_t layers = dbn_t::layers;

    dbn_t& dbn;
    typename dbn_t::tuple_type& tuples;

    sgd_trainer(dbn_t& dbn) : dbn(dbn), tuples(dbn.tuples) {}

    void init_training(std::size_t){}

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
        using namespace etl;

        using rbm_t = RBM;

        static fast_matrix<weight, rbm_t::num_visible, rbm_t::num_hidden> t;

        rbm.w_grad += etl::mmul(reshape<rbm_t::num_visible, 1>(inputs), reshape<1, rbm_t::num_hidden>(rbm.errors), t);

        rbm.b_grad += rbm.errors;
    }

    template<typename T1, typename T2, bool M = rbm_traits<dbn_t>::has_momentum(), enable_if_u<M> = ::detail::dummy>
    T2& get_fgrad(T1& , T2& inc){
        return inc;
    }

    template<typename T1, typename T2, bool M = rbm_traits<dbn_t>::has_momentum(), disable_if_u<M> = ::detail::dummy>
    T1& get_fgrad(T1& grad, T2& ){
        return grad;
    }

    template<typename T, typename L>
    void train_batch(std::size_t /*epoch*/, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        dll_assert(data_batch.size() == label_batch.size(), "Invalid sizes");

        auto n_samples = label_batch.size();

        constexpr const auto n_outputs = dbn_t::template num_hidden<layers - 1>();

        detail::for_each(tuples, [](auto& rbm){
            rbm.w_grad = 0.0;
            rbm.b_grad = 0.0;
            rbm.c_grad = 0.0;

            if(dbn_traits<dbn_t>::has_momentum()){
                rbm.w_inc = 0.0;
                rbm.b_inc = 0.0;
                rbm.c_inc = 0.0;
            }
        });

        //Compute the total gradients for the mini batch

        for(std::size_t i = 0; i < n_samples; ++i){
            //Compute the outputs of each layer one after another
            compute_outputs(data_batch[i]);

            //Compute the errors of the last layer

            auto& last_rbm = dbn.template layer<layers - 1>();

            for(std::size_t j = 0; j < n_outputs; ++j){
                auto observed = last_rbm.o_a[j];
                auto desired = label_batch[i][j];
                last_rbm.errors[j] = observed * (1 - observed) * (desired - observed);
            }

            //Compute the gradients of each layer

            detail::for_each_rpair_i(tuples, [](std::size_t, auto& r1, auto& r2){
                compute_gradients(r2, r1.o_a);

                typedef typename std::remove_reference<decltype(r2)>::type r2_t;

                using namespace etl;

                static fast_matrix<weight, r2_t::num_visible, 1> t;

                r1.errors = r1.o_a * (1 - r1.o_a) * mmul(r2.w, reshape<n_outputs, 1>(r2.errors), t);
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
            //Update momentum gradients
            if(dbn_traits<dbn_t>::has_momentum()){
                auto momentum = dbn.momentum;

                rbm.w_inc = momentum * rbm.w_inc + (1 - momentum) * rbm.w_grad;
                rbm.b_inc = momentum * rbm.b_inc + (1 - momentum) * rbm.b_grad;
                rbm.c_inc = momentum * rbm.c_inc + (1 - momentum) * rbm.c_grad;
            }

            //The final gradients;
            const auto& w_fgrad = get_fgrad(rbm.w_grad, rbm.w_inc);
            const auto& b_fgrad = get_fgrad(rbm.b_grad, rbm.b_inc);
            const auto& c_fgrad = get_fgrad(rbm.c_grad, rbm.c_inc);

            auto learning_rate = dbn.learning_rate;

            rbm.w += learning_rate * w_fgrad;
            rbm.b += learning_rate * b_fgrad;
            rbm.c += learning_rate * c_fgrad;
        });
    }
};

} //end of dbn namespace

#endif