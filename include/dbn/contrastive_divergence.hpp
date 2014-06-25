//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONTRASTIVE_DIVERGENCE_HPP
#define DBN_CONTRASTIVE_DIVERGENCE_HPP

#include "assert.hpp"
#include "batch.hpp"

namespace dbn {

template<typename RBM>
struct base_cd_trainer {
    typedef RBM rbm_t;

    static constexpr const auto num_hidden = rbm_t::num_hidden;
    static constexpr const auto num_visible = rbm_t::num_visible;

    static constexpr const std::size_t num_visible_mom = rbm_t::Momentum ? num_visible : 0;
    static constexpr const std::size_t num_hidden_mom = rbm_t::Momentum ? num_hidden : 0;

    typedef typename rbm_t::weight weight;

    //Gradients
    fast_matrix<weight, num_visible, num_hidden> w_grad;
    fast_vector<weight, num_visible> vbias_grad;
    fast_vector<weight, num_hidden> hbias_grad;

    //Weights and biases for momentum
    fast_matrix<weight, num_visible_mom, num_hidden_mom> w_inc;
    fast_vector<weight, num_visible_mom> a_inc;
    fast_vector<weight, num_hidden_mom> b_inc;

    template<bool M = rbm_t::Momentum, typename std::enable_if<(!M), bool>::type = false>
    base_cd_trainer(){
        static_assert(!rbm_t::Momentum, "This constructor should only be used without momentum support");
    }

    template<bool M = rbm_t::Momentum, typename std::enable_if<(M), bool>::type = false>
    base_cd_trainer() : w_inc(0.0), a_inc(0.0), b_inc(0.0) {
        static_assert(rbm_t::Momentum, "This constructor should only be used with momentum support");
    }

    void update_weights(RBM& rbm){
        auto learning_rate = rbm.learning_rate;

        //Update momentum gradients
        if(rbm_t::Momentum){
            auto momentum = rbm.momentum;

            w_inc = momentum * w_inc + (1 - momentum) * w_grad;
            a_inc = momentum * a_inc + (1 - momentum) * vbias_grad;
            b_inc = momentum * b_inc + (1 - momentum) * hbias_grad;
        }

        //The final gradients;
        const auto& w_fgrad = rbm_t::Momentum ? w_inc : w_grad;
        const auto& a_fgrad = rbm_t::Momentum ? a_inc : vbias_grad;
        const auto& b_fgrad = rbm_t::Momentum ? b_inc : hbias_grad;

        //Weight decay is only applied to weights and not biases
        //Note: According to G. Hinton, Weight Decay should not be applied to
        //biases by default due to their limited number and therefore their weak
        //contribution to overfitting
        //TODO Perhaps this should be configurable

        //Update weights
        if(rbm_t::Decay){
            rbm.w += learning_rate * (w_fgrad - rbm.weight_cost * rbm.w);
        } else {
            rbm.w += learning_rate * w_fgrad;
        }

        //Update visible biases
        rbm.a += learning_rate * a_fgrad;

        //Update hidden biases
        rbm.b += learning_rate * b_fgrad;

        //Check for NaN
        nan_check_3(rbm.w, rbm.a, rbm.b);
    }
};

template<std::size_t K, typename RBM>
struct cd_trainer : base_cd_trainer<RBM> {
private:
    static_assert(K > 0, "CD-0 is not a valid training method");

    typedef RBM rbm_t;
    typedef typename rbm_t::weight weight;

    using base_cd_trainer<RBM>::num_visible;
    using base_cd_trainer<RBM>::num_hidden;

    using base_cd_trainer<RBM>::w_grad;
    using base_cd_trainer<RBM>::vbias_grad;
    using base_cd_trainer<RBM>::hbias_grad;

public:
    cd_trainer() : base_cd_trainer<RBM>() {
        //Nothing else to init here
    }

    template<typename T>
    weight train_batch(const dbn::batch<T>& batch, RBM& rbm){
        dbn_assert(batch.size() <= static_cast<typename dbn::batch<T>::size_type>(BatchSize), "Invalid size");
        dbn_assert(batch[0].size() == num_visible, "The size of the training sample must match visible units");

        //Size of a minibatch
        auto n_samples = static_cast<weight>(batch.size());

        //Clear the gradients
        vbias_grad = 0.0;
        hbias_grad = 0.0;
        w_grad = 0.0;

        for(auto& items : batch){
            rbm.v1 = items;

            //First step
            rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);

            //CD-1
            rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);
            rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);

            //CD-k
            for(std::size_t k = 1; k < K; ++k){
                rbm.activate_visible(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
                rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
            }

            for(size_t i = 0; i < num_visible; ++i){
                for(size_t j = 0; j < num_hidden; ++j){
                    w_grad(i, j) += rbm.h1_a(j) * rbm.v1(i) - rbm.h2_a(j) * rbm.v2_a(i);
                }
            }

            vbias_grad += rbm.v1 - rbm.v2_a;
            hbias_grad += rbm.h1_a - rbm.h2_a;
        }

        //Keep only the mean of the gradients
        w_grad /= n_samples;
        vbias_grad /= n_samples;
        hbias_grad /= n_samples;

        nan_check_3(w_grad, vbias_grad, hbias_grad);

        //Update the weights and biases based on the gradients
        this->update_weights(rbm);

        //Compute the reconstruction error

        weight error = 0.0;
        for(size_t i = 0; i < num_visible; ++i){
            error += vbias_grad(i) * vbias_grad(i);
        }
        error = sqrt(error / num_visible);

        return error;
    }
};

template<std::size_t K, typename RBM>
struct persistent_cd_trainer : base_cd_trainer<RBM> {
private:
    static_assert(K > 0, "PCD-0 is not a valid training method");

    typedef RBM rbm_t;
    typedef typename rbm_t::weight weight;

    using base_cd_trainer<RBM>::num_visible;
    using base_cd_trainer<RBM>::num_hidden;

    using base_cd_trainer<RBM>::w_grad;
    using base_cd_trainer<RBM>::vbias_grad;
    using base_cd_trainer<RBM>::hbias_grad;

    std::vector<fast_vector<weight, num_hidden>> p_h_a;
    std::vector<fast_vector<weight, num_hidden>> p_h_s;

public:
    persistent_cd_trainer() : base_cd_trainer<RBM>() {
        //Nothing else to init here
    }

    template<typename T>
    weight train_batch(const dbn::batch<T>& batch, RBM& rbm){
        dbn_assert(batch.size() <= static_cast<typename dbn::batch<T>::size_type>(rbm_t::BatchSize), "Invalid size");
        dbn_assert(batch[0].size() == num_visible, "The size of the training sample must match visible units");

        //Size of a minibatch
        auto n_samples = static_cast<weight>(batch.size());

        //Clear the gradients
        vbias_grad = 0.0;
        hbias_grad = 0.0;
        w_grad = 0.0;

        bool init = p_h_a.empty();;
        if(init){
            p_h_a.resize(static_cast<typename dbn::batch<T>::size_type>(rbm_t::BatchSize));
            p_h_s.resize(static_cast<typename dbn::batch<T>::size_type>(rbm_t::BatchSize));
        }

        for(std::size_t i = 0; i < batch.size(); ++i){
            auto& items = batch[i];

            rbm.v1 = items;

            //First step
            rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);

            if(init){
                p_h_a[i] = rbm.h1_a;
                p_h_s[i] = rbm.h1_s;
            }

            //CD-1
            rbm.activate_visible(p_h_a[i], p_h_a[i], rbm.v2_a, rbm.v2_s);
            rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);

            //CD-k
            for(std::size_t k = 1; k < K; ++k){
                rbm.activate_visible(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
                rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
            }

            p_h_a[i] = rbm.h2_a;
            p_h_s[i] = rbm.h2_s;

            for(size_t i = 0; i < num_visible; ++i){
                for(size_t j = 0; j < num_hidden; ++j){
                    w_grad(i, j) += rbm.h1_a(j) * rbm.v1(i) - rbm.h2_a(j) * rbm.v2_a(i);
                }
            }

            vbias_grad += rbm.v1 - rbm.v2_a;
            hbias_grad += rbm.h1_a - rbm.h2_a;
        }

        //Keep only the mean of the gradients
        w_grad /= n_samples;
        vbias_grad /= n_samples;
        hbias_grad /= n_samples;

        nan_check_3(w_grad, vbias_grad, hbias_grad);

        //Update the weights and biases based on the gradients
        this->update_weights(rbm);

        //Compute the reconstruction error

        weight error = 0.0;
        for(size_t i = 0; i < num_visible; ++i){
            error += vbias_grad(i) * vbias_grad(i);
        }
        error = sqrt(error / num_visible);

        return error;
    }
};

} //end of dbn namespace

#endif