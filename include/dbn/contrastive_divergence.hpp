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
struct cd1_trainer {
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
    cd1_trainer(){
        static_assert(!rbm_t::Momentum, "This constructor should only be used without momentum support");
    }

    template<bool M = rbm_t::Momentum, typename std::enable_if<(M), bool>::type = false>
    cd1_trainer() : w_inc(0.0), a_inc(0.0), b_inc(0.0) {
        static_assert(rbm_t::Momentum, "This constructor should only be used with momentum support");
    }

    template<typename T>
    double train_batch(const dbn::batch<T>& batch, RBM& rbm){
        dbn_assert(batch.size() <= static_cast<typename dbn::batch<T>::size_type>(BatchSize), "Invalid size");
        dbn_assert(batch[0].size() == num_visible, "The size of the training sample must match visible units");

        //Size of a minibatch
        auto n_samples = static_cast<weight>(batch.size());

        //Clear the gradients
        vbias_grad = 0.0;
        hbias_grad = 0.0;
        w_grad = 0.0;

        rbm.v1 = 0.0;
        rbm.h1_a = 0.0;
        rbm.h1_s = 0.0;
        rbm.h2_a = 0.0;
        rbm.h2_s = 0.0;
        rbm.v2_a = 0.0;
        rbm.v2_s = 0.0;

        for(auto& items : batch){
            rbm.v1 = items;

            rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);
            rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);
            rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);

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

        //Update weights
        if(rbm_t::Momentum){
            if(rbm_t::Decay){
                w_inc = w_inc * rbm.momentum + (w_grad - (rbm.w * rbm.weight_cost)) * rbm.learning_rate;
            } else {
                w_inc = w_inc * rbm.momentum + w_grad * rbm.learning_rate;
            }

            rbm.w += w_inc;
        } else {
            if(rbm_t::Decay){
                rbm.w += (w_grad - (rbm.w * rbm.weight_cost)) * rbm.learning_rate;
            } else {
                rbm.w += w_grad * rbm.learning_rate;
            }
        }

        //Update visible biases
        if(rbm_t::Momentum){
            a_inc = a_inc * rbm.momentum + vbias_grad * rbm.learning_rate;
            rbm.a += a_inc;
        } else {
            rbm.a += vbias_grad * rbm.learning_rate;
        }

        //Update hidden biases
        if(rbm_t::Momentum){
            b_inc = b_inc * rbm.momentum + hbias_grad * rbm.learning_rate;
            rbm.b += b_inc;
        } else {
            rbm.b += hbias_grad * rbm.learning_rate;
        }

        //Check for NaN
        nan_check_3(rbm.w, rbm.a, rbm.b);

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