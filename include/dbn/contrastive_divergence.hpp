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

struct cd1_trainer {
    template<typename T, typename RBM>
    static double train_batch(const dbn::batch<T>& batch, RBM& rbm){
        dbn_assert(batch.size() <= static_cast<typename dbn::batch<T>::size_type>(BatchSize), "Invalid size");
        dbn_assert(batch[0].size() == num_visible, "The size of the training sample must match visible units");
            
        typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;

        //Size of a minibatch
        auto n_samples = static_cast<typename rbm_t::weight>(batch.size());

        constexpr const auto num_hidden = rbm_t::num_hidden;
        constexpr const auto num_visible = rbm_t::num_visible;

        //Clear the gradients
        rbm.vbias_grad = 0.0;
        rbm.hbias_grad = 0.0;
        rbm.w_grad = 0.0;

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
                    rbm.w_grad(i, j) += rbm.h1_a(j) * rbm.v1(i) - rbm.h2_a(j) * rbm.v2_a(i);
                }
            }

            rbm.vbias_grad += rbm.v1 - rbm.v2_a;
            rbm.hbias_grad += rbm.h1_a - rbm.h2_a;
        }

        //Keep only the mean of the gradients
        rbm.w_grad /= n_samples;
        rbm.vbias_grad /= n_samples;
        rbm.hbias_grad /= n_samples;

        nan_check(rbm.w_grad);

        if(rbm_t::Momentum){
            if(rbm_t::Decay){
                rbm.w_inc = rbm.w_inc * rbm.momentum + ((rbm.w_grad / n_samples) - (rbm.w * rbm.weight_cost)) * rbm.learning_rate;
            } else {
                rbm.w_inc = rbm.w_inc * rbm.momentum + rbm.w_grad * (rbm.learning_rate / n_samples);
            }

            rbm.w += rbm.w_inc;
        } else {
            if(rbm_t::Decay){
                rbm.w += ((rbm.w_grad / n_samples) - (rbm.w * rbm.weight_cost)) * rbm.learning_rate;
            } else {
                rbm.w += (rbm.w_grad / n_samples) * rbm.learning_rate;
            }
        }

        nan_check(rbm.w);

        if(rbm_t::Momentum){
            rbm.a_inc = rbm.a_inc * rbm.momentum + (rbm.vbias_grad  / n_samples) * rbm.learning_rate;
            rbm.a += rbm.a_inc;
        } else {
            rbm.a += (rbm.vbias_grad / n_samples) * rbm.learning_rate;
        }

        nan_check(rbm.a);

        if(rbm_t::Momentum){
            rbm.b_inc = rbm.b_inc * rbm.momentum + (rbm.hbias_grad / n_samples) * rbm.learning_rate;
            rbm.b += rbm.b_inc;
        } else {
            rbm.b += (rbm.hbias_grad / n_samples) * rbm.learning_rate;
        }

        nan_check(rbm.b);

        //Compute the reconstruction error

        typename rbm_t::weight error = 0.0;
        for(size_t i = 0; i < num_visible; ++i){
            error += rbm.vbias_grad(i) * rbm.vbias_grad(i);
        }
        error = sqrt((error / (n_samples * n_samples)) / num_visible);

        return error;
    }
};

} //end of dbn namespace

#endif