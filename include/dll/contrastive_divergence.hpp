//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file Contrastive Divergence Implementations
 *
 * Weight decay is applied on biases only on demand (with _FULL variants)
 * Note: According to G. Hinton, Weight Decay should not be applied to biases
 * by default due to their limited number and therefore their weak
 * contribution to overfitting
 */

#pragma once

#include "cpp_utils/assert.hpp"         //Assertions
#include "cpp_utils/maybe_parallel.hpp" //conditional parallel loops
#include "cpp_utils/static_if.hpp"      //static_if for compile-time reduction

#include "etl/etl.hpp"

#include "util/batch.hpp"
#include "util/timers.hpp"
#include "decay_type.hpp"
#include "layer_traits.hpp"
#include "util/blas.hpp"

namespace dll {

#define STATIC_IF_DECAY(d, ...) cpp::static_if<decay == d>([&](auto f) { __VA_ARGS__; });

/*!
 * \brief Base class for all standard trainer
 */
template <typename RBM>
struct base_trainer {
    typedef RBM rbm_t;

    bool init = true;

    template <decay_type decay, typename V, typename G>
    void update_grad(G& grad, const V& value, const RBM& rbm, double penalty) {
        STATIC_IF_DECAY(decay_type::NONE, f(grad) = grad - penalty);
        STATIC_IF_DECAY(decay_type::L1, f(grad) = grad - rbm.l1_weight_cost * abs(value) - penalty);
        STATIC_IF_DECAY(decay_type::L2, f(grad) = grad - rbm.l2_weight_cost * value - penalty);
        STATIC_IF_DECAY(decay_type::L1L2, f(grad) = grad - rbm.l1_weight_cost * abs(value) - rbm.l2_weight_cost * value - penalty);
    }
};

/* The update weights procedure */

template <typename RBM, typename Trainer>
void update_normal(RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:update:normal");

    using rbm_t = RBM;

    //Penalty to be applied to weights and hidden biases
    typename rbm_t::weight w_penalty = 0.0;
    typename rbm_t::weight h_penalty = 0.0;
    typename rbm_t::weight v_penalty = 0.0;

    //Global sparsity method
    cpp::static_if<layer_traits<rbm_t>::sparsity_method() == sparsity_method::GLOBAL_TARGET>([&](auto f) {
        auto decay_rate = rbm.decay_rate;
        auto p          = rbm.sparsity_target;
        auto cost       = rbm.sparsity_cost;

        f(t).q_global_t = decay_rate * t.q_global_t + (1.0 - decay_rate) * t.q_global_batch;

        f(w_penalty) = h_penalty = cost * (t.q_global_t - p);
    });

    //Apply L1/L2 regularization and penalties to the biases

    t.template update_grad<w_decay(layer_traits<rbm_t>::decay())>(t.w_grad, rbm.w, rbm, w_penalty);
    t.template update_grad<b_decay(layer_traits<rbm_t>::decay())>(t.b_grad, rbm.b, rbm, h_penalty);
    t.template update_grad<b_decay(layer_traits<rbm_t>::decay())>(t.c_grad, rbm.c, rbm, v_penalty);

    //Local sparsity method
    cpp::static_if<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET>([&](auto f) {
        auto decay_rate = rbm.decay_rate;
        auto p          = rbm.sparsity_target;
        auto cost       = rbm.sparsity_cost;

        f(t).q_local_t = decay_rate * t.q_local_t + (1.0 - decay_rate) * t.q_local_batch;

        auto q_local_penalty = cost * (t.q_local_t - p);

        f(t).b_grad -= q_local_penalty;

        for (std::size_t i = 0; i < num_hidden(rbm); ++i) {
            for (std::size_t j = 0; j < num_visible(rbm); ++j) {
                f(t).w_grad(j, i) -= q_local_penalty(i);
            }
        }
    });

    //TODO the batch is not necessary full!
    const auto n_samples = double(etl::dim<0>(t.w_grad_b));
    auto eps             = rbm.learning_rate / n_samples;

    //Apply momentum and learning rate
    cpp::static_if<layer_traits<rbm_t>::has_momentum()>([&](auto f) {
        auto momentum = rbm.momentum;

        f(t).w_inc = momentum * t.w_inc + eps * t.w_grad;
        f(t).b_inc = momentum * t.b_inc + eps * t.b_grad;
        f(t).c_inc = momentum * t.c_inc + eps * t.c_grad;

        f(rbm).w += t.w_inc;
        f(rbm).b += t.b_inc;
        f(rbm).c += t.c_inc;
    })
        //Apply the learning rate
        .else_([&](auto f) {
            f(rbm).w += eps * t.w_grad;
            f(rbm).b += eps * t.b_grad;
            f(rbm).c += eps * t.c_grad;
        });

    //Check for NaN
    nan_check_deep_3(rbm.w, rbm.b, rbm.c);
}

template <typename RBM, typename Trainer>
void update_convolutional(RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:update:conv");

    using rbm_t  = RBM;
    using weight = typename rbm_t::weight;

    //Penalty to be applied to weights and hidden biases
    weight w_penalty = 0.0;
    weight h_penalty = 0.0;
    weight v_penalty = 0.0;

    //Global sparsity method
    cpp::static_if<layer_traits<rbm_t>::sparsity_method() == sparsity_method::GLOBAL_TARGET>([&](auto f) {
        auto decay_rate = rbm.decay_rate;
        auto p          = rbm.sparsity_target;
        auto cost       = rbm.sparsity_cost;

        f(t).q_global_t = decay_rate * t.q_global_t + (1.0 - decay_rate) * t.q_global_batch;

        f(w_penalty) = h_penalty = cost * (t.q_global_t - p);
    });

    //Apply L1/L2 regularization and penalties to the biases

    t.template update_grad<w_decay(layer_traits<rbm_t>::decay())>(t.w_grad, rbm.w, rbm, w_penalty);
    t.template update_grad<b_decay(layer_traits<rbm_t>::decay())>(t.b_grad, rbm.b, rbm, h_penalty);
    t.template update_grad<b_decay(layer_traits<rbm_t>::decay())>(t.c_grad, rbm.c, rbm, v_penalty);

    //Local sparsity method
    cpp::static_if<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET>([&](auto f) {
        auto decay_rate = rbm.decay_rate;
        auto p          = rbm.sparsity_target;
        auto cost       = rbm.sparsity_cost;

        f(t).q_local_t = decay_rate * t.q_local_t + (1.0 - decay_rate) * t.q_local_batch;

        auto q_local_penalty = cost * (t.q_local_t - p);

        f(t).b_grad -= sum_r(q_local_penalty);

        const auto K   = get_k(rbm);

        auto k_penalty = sum_r(q_local_penalty);
        for (std::size_t k = 0; k < K; ++k) {
            f(t).w_grad(k) = t.w_grad(k) - k_penalty(k);
        }
    });

    //Honglak Lee's sparsity method
    cpp::static_if<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE>([&](auto f) {
        f(t).w_grad -= rbm.pbias_lambda * t.w_bias;
        f(t).b_grad -= rbm.pbias_lambda * t.b_bias;
        f(t).c_grad -= rbm.pbias_lambda * t.c_bias;
    });

    const auto n_samples = get_batch_size(rbm);
    auto eps             = rbm.learning_rate / n_samples;

    //Apply momentum and learning rate
    cpp::static_if<layer_traits<rbm_t>::has_momentum()>([&](auto f) {
        auto momentum = rbm.momentum;

        f(t).w_inc = momentum * t.w_inc + eps * t.w_grad;
        f(t).b_inc = momentum * t.b_inc + eps * t.b_grad;
        f(t).c_inc = momentum * t.c_inc + eps * t.c_grad;

        f(rbm.w) += t.w_inc;
        f(rbm.b) += t.b_inc;
        f(rbm.c) += t.c_inc;
    })
    //Apply learning rate only
    .else_([&](auto f) {
        f(rbm.w) += eps * t.w_grad;
        f(rbm.b) += eps * t.b_grad;
        f(rbm.c) += eps * t.c_grad;
    });

    //Check for NaN
    nan_check_deep(rbm.w);
    nan_check_deep(rbm.b);
    nan_check_deep(rbm.c);
}

#ifndef ETL_BLAS_MODE

template <typename Trainer>
void batch_compute_gradients(Trainer& t) {
    dll::auto_timer timer("cd:batch_compute_gradients:std");

    const auto B = etl::dim<0>(t.w_grad_b);
    const auto NV = etl::dim<1>(t.w_grad_b);
    const auto NH = etl::dim<2>(t.w_grad_b);

    for (std::size_t b = 0; b < B; b++) {
        for (std::size_t i = 0; i < NV; i++) {
            for (std::size_t j = 0; j < NH; j++) {
                t.w_grad(i, j) += t.vf(b, i) * t.h1_a(b, j) - t.v2_a(b, i) * t.h2_a(b, j);
            }
        }
    }

    for (std::size_t b = 0; b < B; b++) {
        for (std::size_t i = 0; i < NH; i++) {
            t.b_grad(i) += t.h1_a(b, i) - t.h2_a(b, i);
        }
    }

    for (std::size_t b = 0; b < B; b++) {
        for (std::size_t i = 0; i < NV; i++) {
            t.c_grad(i) += t.vf(b, i) - t.v2_a(b, i);
        }
    }
}

template <typename Trainer>
void compute_gradients_one(Trainer& t) {
    dll::auto_timer timer("cd:compute_gradients_one:std");

    t.w_grad = 0;

    for (std::size_t i = 0; i < etl::dim<0>(t.w_grad); i++) {
        for (std::size_t j = 0; j < etl::dim<1>(t.w_grad); j++) {
            t.w_grad(i, j) += t.vf(0, i) * t.h1_a(0, j) - t.v2_a(0, i) * t.h2_a(0, j);
        }
    }

    t.b_grad = t.h1_a(0) - t.h2_a(0);
    t.c_grad = t.vf(0) - t.v2_a(0);
}

#else

template <typename Trainer>
void batch_compute_gradients(Trainer& t) {
    dll::auto_timer timer("cd:batch_compute_gradients:blas");

    const auto B = etl::dim<0>(t.w_grad_b);

    for (std::size_t b = 0; b < B; b++) {
        blas_ger(
            etl::dim<1>(t.vf), etl::dim<1>(t.h1_a),
            1.0,
            t.vf(b).memory_start(),
            t.h1_a(b).memory_start(),
            t.w_grad.memory_start());

        blas_ger(
            etl::dim<1>(t.v2_a), etl::dim<1>(t.h2_a),
            -1.0,
            t.v2_a(b).memory_start(),
            t.h2_a(b).memory_start(),
            t.w_grad.memory_start());
    }

    for (std::size_t b = 0; b < B; b++) {
        blas_axpy(etl::dim<1>(t.h1_a), 1.0, t.h1_a(b).memory_start(), t.b_grad.memory_start());
        blas_axpy(etl::dim<1>(t.h2_a), -1.0, t.h2_a(b).memory_start(), t.b_grad.memory_start());
    }

    for (std::size_t b = 0; b < B; b++) {
        blas_axpy(etl::dim<1>(t.vf), 1.0, t.vf(b).memory_start(), t.c_grad.memory_start());
        blas_axpy(etl::dim<1>(t.v2_a), -1.0, t.v2_a(b).memory_start(), t.c_grad.memory_start());
    }
}

template <typename Trainer>
void compute_gradients_one(Trainer& t) {
    dll::auto_timer timer("cd:compute_gradients_one:blas");

    t.w_grad = 0;
    t.b_grad = 0;
    t.c_grad = 0;

    blas_ger(
        etl::dim<1>(t.vf), etl::dim<1>(t.h1_a),
        1.0,
        t.vf(0).memory_start(),
        t.h1_a(0).memory_start(),
        t.w_grad.memory_start());

    blas_ger(
        etl::dim<1>(t.v2_a), etl::dim<1>(t.h2_a),
        -1.0,
        t.v2_a(0).memory_start(),
        t.h2_a(0).memory_start(),
        t.w_grad.memory_start());

    blas_axpy(etl::dim<1>(t.h1_a), 1.0, t.h1_a(0).memory_start(), t.b_grad.memory_start());
    blas_axpy(etl::dim<1>(t.h2_a), -1.0, t.h2_a(0).memory_start(), t.b_grad.memory_start());

    blas_axpy(etl::dim<1>(t.vf), 1.0, t.vf(0).memory_start(), t.c_grad.memory_start());
    blas_axpy(etl::dim<1>(t.v2_a), -1.0, t.v2_a(0).memory_start(), t.c_grad.memory_start());
}

#endif

/* The training procedures */

template <bool Persistent, std::size_t K, typename T, typename RBM, typename Trainer, cpp_enable_if(layer_traits<RBM>::is_parallel_mode())>
void compute_gradients_normal(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:gradients:normal:par");

    auto n = input_batch.size();

    // clang-format off
    maybe_parallel_foreach_pair_i(t.pool, input_batch.begin(), input_batch.end(), expected_batch.begin(), expected_batch.end(),
            [&](const auto& input, const auto& expected, std::size_t i)
    {
        //Copy input/expected for computations
        t.v1(i) = input;
        t.vf(i) = expected;

        //First step
        rbm.template activate_hidden<true, true>(t.h1_a(i), t.h1_s(i), t.v1(i), t.v1(i));

        if(Persistent && t.init){
            t.p_h_a(i) = t.h1_a(i);
            t.p_h_s(i) = t.h1_s(i);
        }

        //CD-1
        cpp::static_if<Persistent>([&](auto f){
            f(rbm).template activate_visible<true, false>(t.p_h_a(i), t.p_h_s(i), t.v2_a(i), t.v2_s(i));
            f(rbm).template activate_hidden<true, true>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i));
        }).else_([&](auto f){
            f(rbm).template activate_visible<true, false>(t.h1_a(i), t.h1_s(i), t.v2_a(i), t.v2_s(i));
            f(rbm).template activate_hidden<true, (K > 1)>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i));
        });

        //CD-k
        for(std::size_t k = 1; k < K; ++k){
            rbm.template activate_visible<true, false>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i));
            rbm.template activate_hidden<true, true>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i));
        }

        if(n > 1){
            //Reset the batch gradients
            t.w_grad_b(i) = 0;

            for(std::size_t i2 = 0; i2 < num_visible(rbm); i2++){
                for(std::size_t j = 0; j < num_hidden(rbm); j++){
                    t.w_grad_b(i, i2, j) += t.vf(i, i2) * t.h1_a(i,j) - t.v2_a(i, i2) * t.h2_a(i, j);
                }
            }
        } else {
            compute_gradients_one(t);
        }
    });
    // clang-format on

    if(n > 1){
        //Compute the gradients
        t.w_grad = sum_l(t.w_grad_b);
        t.b_grad = sum_l(t.h1_a - t.h2_a);
        t.c_grad = sum_l(t.vf - t.v2_a);
    }
}

template <bool Persistent, std::size_t K, typename T, typename RBM, typename Trainer, cpp_disable_if(layer_traits<RBM>::is_parallel_mode())>
void compute_gradients_normal(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:gradients:normal:batch");

    //Copy input/expected for computations
    auto iit  = input_batch.begin();
    auto iend = input_batch.end();
    auto eit  = expected_batch.begin();

    for (std::size_t i = 0; iit != iend; ++i, ++iit, ++eit) {
        t.v1(i) = *iit;
        t.vf(i) = *eit;
    }

    //First step
    rbm.template batch_activate_hidden<true, true>(t.h1_a, t.h1_s, t.v1, t.v1);

    if (Persistent && t.init) {
        t.p_h_a = t.h1_a;
        t.p_h_s = t.h1_s;
    }

    //CD-1
    cpp::static_if<Persistent>([&](auto f) {
        f(rbm).template batch_activate_visible<true, false>(t.p_h_a, t.p_h_s, t.v2_a, t.v2_s);
        f(rbm).template batch_activate_hidden<true, true>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
    }).else_([&](auto f) {
        f(rbm).template batch_activate_visible<true, false>(t.h1_a, t.h1_s, t.v2_a, t.v2_s);
        f(rbm).template batch_activate_hidden<true, (K > 1)>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
    });

    //CD-k
    for (std::size_t k = 1; k < K; ++k) {
        rbm.template batch_activate_visible<true, false>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
        rbm.template batch_activate_hidden<true, true>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
    }

    //Compute the gradients

    t.w_grad = 0;
    t.b_grad = 0;
    t.c_grad = 0;

    batch_compute_gradients(t);
}

template <bool Persistent, std::size_t K, typename T, typename RBM, typename Trainer>
void train_normal(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context, RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:train:normal");

    cpp_assert(input_batch.size() > 0, "Invalid batch size");
    cpp_assert(input_batch.size() <= get_batch_size(rbm), "Invalid batch size");
    cpp_assert(input_batch.begin()->size() == input_size(rbm), "The size of the training sample must match visible units");

    using namespace etl;
    using rbm_t = RBM;

    compute_gradients_normal<Persistent, K>(input_batch, expected_batch, rbm, t);

    if (Persistent) {
        t.p_h_a = t.h2_a;
        t.p_h_s = t.h2_s;

        t.init = false;
    }

    context.batch_error = mean((t.vf - t.v2_a) >> (t.vf - t.v2_a));

    nan_check_deep_3(t.w_grad, t.b_grad, t.c_grad);

    //Compute the mean activation probabilities
    t.q_global_batch = mean(t.h2_a);

    cpp::static_if<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET>([&](auto f) {
        f(t).q_local_batch = mean_l(t.h2_a);
    });

    context.batch_sparsity = t.q_global_batch;

    //Update the weights and biases based on the gradients
    t.update(rbm);
}

template <bool Denoising, typename Trainer, typename RBM>
void normal_compute_gradients_conv(RBM& /*rbm*/, Trainer& t) {
    dll::auto_timer timer("cd:normal_compute_gradients_conv");

    using namespace etl;

    if (Denoising) {
        t.w_pos = etl::conv_4d_valid_filter_flipped(t.vf, t.h1_a);
        t.w_neg = etl::conv_4d_valid_filter_flipped(t.v2_a, t.h2_a);
    } else {
        t.w_pos = etl::conv_4d_valid_filter_flipped(t.v1, t.h1_a);
        t.w_neg = etl::conv_4d_valid_filter_flipped(t.v2_a, t.h2_a);
    }
}

template <bool Persistent, bool Denoising, std::size_t N, typename Trainer, typename T, typename RBM, cpp_enable_if(layer_traits<RBM>::is_parallel_mode())>
void compute_gradients_conv(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:gradients:conv:par");

    // clang-format off
    maybe_parallel_foreach_pair_i(t.pool, input_batch.begin(), input_batch.end(), expected_batch.begin(), expected_batch.end(),
            [&](const auto& input, const auto& expected, std::size_t i)
    {
        //Copy input/expected for computations
        t.v1(i) = input;

        if(Denoising){
            t.vf(i) = expected;
        }

        //First step
        rbm.template activate_hidden<true, true>(t.h1_a(i), t.h1_s(i), t.v1(i), t.v1(i), t.v_cv(i));

        if(Persistent && t.init){
            t.p_h_a(i) = t.h1_a(i);
            t.p_h_s(i) = t.h1_s(i);
        }

        //CD-1
        if(Persistent){
            rbm.template activate_visible<true, false>(t.p_h_a(i), t.p_h_s(i), t.v2_a(i), t.v2_s(i), t.h_cv(i));
            rbm.template activate_hidden<true, true>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.v_cv(i));
        } else {
            rbm.template activate_visible<true, false>(t.h1_a(i), t.h1_s(i), t.v2_a(i), t.v2_s(i), t.h_cv(i));
            rbm.template activate_hidden<true, (N > 1)>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.v_cv(i));
        }

        //CD-k
        for(std::size_t k = 1; k < N; ++k){
            rbm.template activate_visible<true, false>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.h_cv(i));
            rbm.template activate_hidden<true, true>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.v_cv(i));
        }
    });
    // clang-format on

    //Compute gradients
    normal_compute_gradients_conv<Denoising>(rbm, t);
}

template <bool Denoising, typename Trainer, typename RBM>
void batch_compute_gradients_conv(RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:batch_compute_gradients_conv");

    using namespace etl;

    if (Denoising) {
        t.w_pos = etl::conv_4d_valid_filter_flipped(t.vf, t.h1_a);
        t.w_neg = etl::conv_4d_valid_filter_flipped(t.v2_a, t.h2_a);
    } else {
        t.w_pos = etl::conv_4d_valid_filter_flipped(t.v1, t.h1_a);
        t.w_neg = etl::conv_4d_valid_filter_flipped(t.v2_a, t.h2_a);
    }

    cpp_unused(rbm);
}

template <bool Persistent, bool Denoising, std::size_t N, typename Trainer, typename T, typename RBM, cpp_disable_if(layer_traits<RBM>::is_parallel_mode())>
void compute_gradients_conv(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:gradients:conv:batch");

    //Copy input/expected for computations
    auto iit  = input_batch.begin();
    auto iend = input_batch.end();

    for (std::size_t i = 0; iit != iend; ++i, ++iit) {
        t.v1(i) = *iit;
    }

    if (Denoising) {
        auto eit  = expected_batch.begin();
        auto eend = expected_batch.end();

        for (std::size_t i = 0; eit != eend; ++i, ++eit) {
            t.vf(i) = *eit;
        }
    }

    //First step
    rbm.template batch_activate_hidden<true, true>(t.h1_a, t.h1_s, t.v1, t.v1, t.v_cv);

    if (Persistent && t.init) {
        t.p_h_a = t.h1_a;
        t.p_h_s = t.h1_s;
    }

    //CD-1
    if (Persistent) {
        rbm.template batch_activate_visible<true, false>(t.p_h_a, t.p_h_s, t.v2_a, t.v2_s, t.h_cv);
        rbm.template batch_activate_hidden<true, true>(t.h2_a, t.h2_s, t.v2_a, t.v2_s, t.v_cv);
    } else {
        rbm.template batch_activate_visible<true, false>(t.h1_a, t.h1_s, t.v2_a, t.v2_s, t.h_cv);
        rbm.template batch_activate_hidden<true, (N > 1)>(t.h2_a, t.h2_s, t.v2_a, t.v2_s, t.v_cv);
    }

    //CD-k
    for (std::size_t k = 1; k < N; ++k) {
        rbm.template batch_activate_visible<true, false>(t.h2_a, t.h2_s, t.v2_a, t.v2_s, t.h_cv);
        rbm.template batch_activate_hidden<true, true>(t.h2_a, t.h2_s, t.v2_a, t.v2_s, t.v_cv);
    }

    //Compute gradients
    batch_compute_gradients_conv<Denoising>(rbm, t);
}

template <bool Persistent, bool Denoising, std::size_t N, typename Trainer, typename T, typename RBM>
void train_convolutional(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context, RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:train:conv");

    cpp_assert(input_batch.size() > 0, "Invalid batch size");
    cpp_assert(input_batch.size() <= get_batch_size(rbm), "Invalid batch size");
    cpp_assert(input_batch.size() == expected_batch.size(), "Batches do not match");
    cpp_assert(input_batch.begin()->size() == input_size(rbm), "The size of the training sample must match visible units");

    using rbm_t = RBM;

    compute_gradients_conv<Persistent, Denoising, N>(input_batch, expected_batch, rbm, t);

    if (Persistent) {
        t.p_h_a = t.h2_a;
        t.p_h_s = t.h2_s;

        t.init = false;
    }

    //Compute the gradients
    t.w_grad = t.w_pos - t.w_neg;

    t.b_grad = mean_r(sum_l(t.h1_a - t.h2_a));

    cpp::static_if<Denoising>([&](auto f) {
        f(t).c_grad = mean_r(sum_l(t.vf - t.v2_a));
    }).else_([&](auto f) { f(t).c_grad = mean_r(sum_l(t.v1 - t.v2_a)); });

    nan_check_deep(t.w_grad);
    nan_check_deep(t.b_grad);
    nan_check_deep(t.c_grad);

    //Compute the mean activation probabilities
    t.q_global_batch = mean(t.h2_a);

    cpp::static_if<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET>([&](auto f) {
        f(t).q_local_batch = mean_l(t.h2_a);
    });

    //Compute the biases for sparsity

    //Only b_bias are supported for now
    cpp::static_if<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE && layer_traits<rbm_t>::bias_mode() == bias_mode::SIMPLE>([&](auto f) {
        f(t).b_bias = mean_r(mean_l(t.h2_a)) - rbm.pbias;
    });

    //Accumulate the sparsity
    context.batch_sparsity = t.q_global_batch;

    //Accumulate the error
    cpp::static_if<Denoising>([&](auto f) {
        f(context).batch_error = mean(etl::scale((t.vf - t.v2_a), (t.vf - t.v2_a)));
    }).else_([&](auto f) { f(context).batch_error = mean(etl::scale((t.v1 - t.v2_a), (t.v1 - t.v2_a))); });

    //Update the weights and biases based on the gradients
    t.update(rbm);
}

/* The specialized trainers */

/*!
 * \brief Base class for all Contrastive Divergence Trainer.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template <std::size_t N, typename RBM, bool Persistent, bool Denoising, typename Enable = void>
struct base_cd_trainer : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    using rbm_t  = RBM;
    using weight = typename rbm_t::weight;

    static constexpr const auto num_hidden  = rbm_t::num_hidden;
    static constexpr const auto num_visible = rbm_t::num_visible;

    static constexpr const auto batch_size = layer_traits<rbm_t>::batch_size();

    rbm_t& rbm;

    etl::fast_matrix<weight, batch_size, num_visible> v1; ///< Input
    etl::fast_matrix<weight, batch_size, num_visible> vf; ///< Expected

    etl::fast_matrix<weight, batch_size, num_hidden> h1_a;
    etl::fast_matrix<weight, batch_size, num_hidden> h1_s;

    etl::fast_matrix<weight, batch_size, num_visible> v2_a;
    etl::fast_matrix<weight, batch_size, num_visible> v2_s;

    etl::fast_matrix<weight, batch_size, num_hidden> h2_a;
    etl::fast_matrix<weight, batch_size, num_hidden> h2_s;

    etl::fast_matrix<weight, batch_size, num_visible, num_hidden> w_grad_b;

    //Gradients
    etl::fast_matrix<weight, num_visible, num_hidden> w_grad;
    etl::fast_vector<weight, num_hidden> b_grad;
    etl::fast_vector<weight, num_visible> c_grad;

    //{{{ Momentum

    etl::fast_matrix<weight, num_visible, num_hidden> w_inc;
    etl::fast_vector<weight, num_hidden> b_inc;
    etl::fast_vector<weight, num_visible> c_inc;

    //}}} Momentum end

    //{{{ Sparsity

    weight q_global_batch;
    weight q_global_t;

    etl::fast_matrix<weight, num_hidden> q_local_batch;
    etl::fast_vector<weight, num_hidden> q_local_t;

    //}}} Sparsity end

    etl::fast_matrix<weight, batch_size, rbm_t::num_hidden> p_h_a;
    etl::fast_matrix<weight, batch_size, rbm_t::num_hidden> p_h_s;

    cpp::thread_pool<!layer_traits<rbm_t>::is_serial()> pool;

    template <bool M = layer_traits<rbm_t>::has_momentum(), cpp_disable_if(M)>
    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm), q_global_t(0.0), q_local_t(0.0), pool(etl::threads) {
        static_assert(!layer_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template <bool M = layer_traits<rbm_t>::has_momentum(), cpp_enable_if(M)>
    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm), w_inc(0.0), b_inc(0.0), c_inc(0.0), q_global_t(0.0), q_local_t(0.0), pool(etl::threads) {
        static_assert(layer_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update(RBM& rbm) {
        update_normal(rbm, *this);
    }

    template <typename T>
    void train_batch(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context) {
        train_normal<Persistent, N>(input_batch, expected_batch, context, rbm, *this);
    }

    static std::string name() {
        return std::string("") + (Persistent ? "Persistent " : "") + (Denoising ? "Denoising " : "") + "Contrastive Divergence";
    }
};

/*!
 * \brief Base class for all Contrastive Divergence Trainer.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template <std::size_t N, typename RBM, bool Persistent, bool Denoising>
struct base_cd_trainer<N, RBM, Persistent, Denoising, std::enable_if_t<layer_traits<RBM>::is_dynamic() && !layer_traits<RBM>::is_convolutional_rbm_layer()>> : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    typedef RBM rbm_t;

    typedef typename rbm_t::weight weight;

    rbm_t& rbm;

    etl::dyn_matrix<weight> v1; ///< Input
    etl::dyn_matrix<weight> vf; ///< Expected

    etl::dyn_matrix<weight> h1_a;
    etl::dyn_matrix<weight> h1_s;

    etl::dyn_matrix<weight> v2_a;
    etl::dyn_matrix<weight> v2_s;

    etl::dyn_matrix<weight> h2_a;
    etl::dyn_matrix<weight> h2_s;

    etl::dyn_matrix<weight, 3> w_grad_b;

    //Gradients
    etl::dyn_matrix<weight> w_grad;
    etl::dyn_vector<weight> b_grad;
    etl::dyn_vector<weight> c_grad;

    //{{{ Momentum

    etl::dyn_matrix<weight> w_inc;
    etl::dyn_vector<weight> b_inc;
    etl::dyn_vector<weight> c_inc;

    //}}} Momentum end

    //{{{ Sparsity

    weight q_global_batch;
    weight q_global_t;

    etl::dyn_vector<weight> q_local_batch;
    etl::dyn_vector<weight> q_local_t;

    //}}} Sparsity end

    etl::dyn_matrix<weight> p_h_a;
    etl::dyn_matrix<weight> p_h_s;

    cpp::thread_pool<!layer_traits<rbm_t>::is_serial()> pool;

    template <bool M = layer_traits<rbm_t>::has_momentum(), cpp_disable_if(M)>
    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm),
              v1(get_batch_size(rbm), rbm.num_visible),
              vf(get_batch_size(rbm), rbm.num_visible),
              h1_a(get_batch_size(rbm), rbm.num_hidden),
              h1_s(get_batch_size(rbm), rbm.num_hidden),
              v2_a(get_batch_size(rbm), rbm.num_visible),
              v2_s(get_batch_size(rbm), rbm.num_visible),
              h2_a(get_batch_size(rbm), rbm.num_hidden),
              h2_s(get_batch_size(rbm), rbm.num_hidden),
              w_grad_b(get_batch_size(rbm), rbm.num_visible, rbm.num_hidden),
              w_grad(rbm.num_visible, rbm.num_hidden),
              b_grad(rbm.num_hidden),
              c_grad(rbm.num_visible),
              w_inc(0, 0),
              b_inc(0),
              c_inc(0),
              q_global_t(0.0),
              q_local_batch(rbm.num_hidden),
              q_local_t(rbm.num_hidden, static_cast<weight>(0.0)),
              p_h_a(get_batch_size(rbm), rbm.num_hidden),
              p_h_s(get_batch_size(rbm), rbm.num_hidden), pool(etl::threads) {
        static_assert(!layer_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template <bool M = layer_traits<rbm_t>::has_momentum(), cpp_enable_if(M)>
    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm),
              v1(get_batch_size(rbm), rbm.num_visible),
              vf(get_batch_size(rbm), rbm.num_visible),
              h1_a(get_batch_size(rbm), rbm.num_hidden),
              h1_s(get_batch_size(rbm), rbm.num_hidden),
              v2_a(get_batch_size(rbm), rbm.num_visible),
              v2_s(get_batch_size(rbm), rbm.num_visible),
              h2_a(get_batch_size(rbm), rbm.num_hidden),
              h2_s(get_batch_size(rbm), rbm.num_hidden),
              w_grad_b(get_batch_size(rbm), rbm.num_visible, rbm.num_hidden),
              w_grad(rbm.num_visible, rbm.num_hidden),
              b_grad(rbm.num_hidden),
              c_grad(rbm.num_visible),
              w_inc(rbm.num_visible, rbm.num_hidden, static_cast<weight>(0.0)),
              b_inc(rbm.num_hidden, static_cast<weight>(0.0)),
              c_inc(rbm.num_visible, static_cast<weight>(0.0)),
              q_global_t(0.0),
              q_local_batch(rbm.num_hidden),
              q_local_t(rbm.num_hidden, static_cast<weight>(0.0)),
              p_h_a(get_batch_size(rbm), rbm.num_hidden),
              p_h_s(get_batch_size(rbm), rbm.num_hidden), pool(etl::threads) {
        static_assert(layer_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update(RBM& rbm) {
        update_normal(rbm, *this);
    }

    template <typename T>
    void train_batch(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context) {
        train_normal<Persistent, N>(input_batch, expected_batch, context, rbm, *this);
    }

    static std::string name() {
        return std::string("") + (Persistent ? "Persistent " : "") + (Denoising ? "Denoising " : "") + "Contrastive Divergence (dynamic)";
    }
};

/*!
 * \brief Specialization of base_cd_trainer for Convolutional RBM.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template <std::size_t N, typename RBM, bool Persistent, bool Denoising>
struct base_cd_trainer<N, RBM, Persistent, Denoising, std::enable_if_t<!layer_traits<RBM>::is_dynamic() && layer_traits<RBM>::is_convolutional_rbm_layer()>> : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    using rbm_t = RBM;

    static constexpr const auto K   = rbm_t::K;
    static constexpr const auto NC  = rbm_t::NC;
    static constexpr const auto NV1 = rbm_t::NV1;
    static constexpr const auto NV2 = rbm_t::NV2;
    static constexpr const auto NH1 = rbm_t::NH1;
    static constexpr const auto NH2 = rbm_t::NH2;
    static constexpr const auto NW1 = rbm_t::NW1;
    static constexpr const auto NW2 = rbm_t::NW2;

    static constexpr const auto batch_size = layer_traits<rbm_t>::batch_size();

    typedef typename rbm_t::weight weight;

    rbm_t& rbm;

#define W_DIMS K, NC, NW1, NW2

    //Gradients
    etl::fast_matrix<weight, W_DIMS> w_grad; //Gradients of shared weights
    etl::fast_vector<weight, K> b_grad;               //Gradients of hidden biases bk
    etl::fast_vector<weight, NC> c_grad;              //Visible gradient

    //{{{ Momentum

    etl::fast_matrix<weight, W_DIMS> w_inc;
    etl::fast_vector<weight, K> b_inc;
    etl::fast_vector<weight, NC> c_inc;

    //}}} Momentum end

    //{{{ Sparsity

    weight q_global_batch;
    weight q_global_t;

    conditional_fast_matrix_t<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET, weight, K, NH1, NH2> q_local_batch;
    conditional_fast_matrix_t<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET, weight, K, NH1, NH2> q_local_t;

    //}}} Sparsity end

    //{{{ Sparsity biases

    conditional_fast_matrix_t<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE, weight, W_DIMS> w_bias;
    conditional_fast_matrix_t<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE, weight, K> b_bias;
    conditional_fast_matrix_t<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE, weight, NC> c_bias;

    //}}} Sparsity biases end

    static constexpr const std::size_t V_CV_CHANNELS = 2;
    static constexpr const std::size_t H_CV_CHANNELS = 2;

    etl::fast_matrix<weight, batch_size, V_CV_CHANNELS, K, NH1, NH2> v_cv;
    etl::fast_matrix<weight, batch_size, H_CV_CHANNELS, NV1, NV2> h_cv;

    conditional_fast_matrix_t<Persistent, weight, batch_size, K, NH1, NH2> p_h_a;
    conditional_fast_matrix_t<Persistent, weight, batch_size, K, NH1, NH2> p_h_s;

    etl::fast_matrix<weight, W_DIMS> w_pos;
    etl::fast_matrix<weight, W_DIMS> w_neg;

    etl::fast_matrix<weight, batch_size, NC, NV1, NV2> v1;                     //Input
    conditional_fast_matrix_t<Denoising, weight, batch_size, NC, NV1, NV2> vf; //Expected

    etl::fast_matrix<weight, batch_size, K, NH1, NH2> h1_a;
    etl::fast_matrix<weight, batch_size, K, NH1, NH2> h1_s;

    etl::fast_matrix<weight, batch_size, NC, NV1, NV2> v2_a;
    conditional_fast_matrix_t<false, weight, batch_size, NC, NV1, NV2> v2_s;

    etl::fast_matrix<weight, batch_size, K, NH1, NH2> h2_a;
    conditional_fast_matrix_t<(N > 1), weight, batch_size, K, NH1, NH2> h2_s;

    cpp::thread_pool<!layer_traits<rbm_t>::is_serial()> pool;

    template <bool M = layer_traits<rbm_t>::has_momentum(), cpp_disable_if(M)>
    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm),
              q_global_t(0.0),
              q_local_t(0.0),
              w_bias(0.0),
              b_bias(0.0),
              c_bias(0.0), pool(etl::threads) {
        static_assert(!layer_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template <bool M = layer_traits<rbm_t>::has_momentum(), cpp_enable_if(M)>
    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm),
              w_inc(0.0),
              b_inc(0.0),
              c_inc(0.0),
              q_global_t(0.0),
              q_local_t(0.0),
              w_bias(0.0),
              b_bias(0.0),
              c_bias(0.0), pool(etl::threads) {
        static_assert(layer_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update(RBM& rbm) {
        update_convolutional(rbm, *this);
    }

    template <typename T>
    void train_batch(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context) {
        train_convolutional<Persistent, Denoising, N>(input_batch, expected_batch, context, rbm, *this);
    }

    static std::string name() {
        return std::string("") + (Persistent ? "Persistent " : "") + (Denoising ? "Denoising " : "") + "Contrastive Divergence (convolutional)";
    }
};

/*!
 * \brief Specialization of base_cd_trainer for dynamic Convolutional RBM.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template <std::size_t N, typename RBM, bool Persistent, bool Denoising>
struct base_cd_trainer<N, RBM, Persistent, Denoising, std::enable_if_t<layer_traits<RBM>::is_dynamic() && layer_traits<RBM>::is_convolutional_rbm_layer()>> : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    using rbm_t = RBM;

    typedef typename rbm_t::weight weight;

    rbm_t& rbm;

#define DYN_W_DIMS rbm.k, rbm.nc, rbm.nw1, rbm.nw2

    //Gradients
    etl::dyn_matrix<weight, 4> w_grad; //Gradients of shared weights
    etl::dyn_matrix<weight, 1> b_grad;               //Gradients of hidden biases bk
    etl::dyn_matrix<weight, 1> c_grad;              //Visible gradient

    //{{{ Momentum

    etl::dyn_matrix<weight, 4> w_inc;
    etl::dyn_matrix<weight, 1> b_inc;
    etl::dyn_matrix<weight, 1> c_inc;

    //}}} Momentum end

    //{{{ Sparsity

    weight q_global_batch;
    weight q_global_t;

    etl::dyn_matrix<weight, 3> q_local_batch;
    etl::dyn_matrix<weight, 3> q_local_t;

    //}}} Sparsity end

    //{{{ Sparsity biases

    etl::dyn_matrix<weight, 4> w_bias;
    etl::dyn_matrix<weight, 1> b_bias;
    etl::dyn_matrix<weight, 1> c_bias;

    //}}} Sparsity biases end

    static constexpr const std::size_t V_CV_CHANNELS = 2;
    static constexpr const std::size_t H_CV_CHANNELS = 2;

    etl::dyn_matrix<weight, 5> v_cv;
    etl::dyn_matrix<weight, 4> h_cv;

    etl::dyn_matrix<weight, 4> p_h_a;
    etl::dyn_matrix<weight, 4> p_h_s;

    etl::dyn_matrix<weight, 4> w_pos;
    etl::dyn_matrix<weight, 4> w_neg;

    etl::dyn_matrix<weight, 4> v1;                     //Input
    etl::dyn_matrix<weight, 4> vf; //Expected

    etl::dyn_matrix<weight, 4> h1_a;
    etl::dyn_matrix<weight, 4> h1_s;

    etl::dyn_matrix<weight, 4> v2_a;
    etl::dyn_matrix<weight, 4> v2_s;

    etl::dyn_matrix<weight, 4> h2_a;
    etl::dyn_matrix<weight, 4> h2_s;

    cpp::thread_pool<!layer_traits<rbm_t>::is_serial()> pool;

    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm),
             w_grad(DYN_W_DIMS, 0.0),
             b_grad(rbm.k, 0.0),
             c_grad(rbm.nc, 0.0),
             w_inc(DYN_W_DIMS, 0.0),
             b_inc(rbm.k, 0.0),
             c_inc(rbm.nc, 0.0),
             q_global_t(0.0),
             q_local_batch(rbm.k, rbm.nh1, rbm.nh2),
             q_local_t(rbm.k, rbm.nh1, rbm.nh2, 0.0),
             w_bias(DYN_W_DIMS, 0.0),
             b_bias(rbm.k, 0.0),
             c_bias(rbm.nc, 0.0),
             v_cv(get_batch_size(rbm), V_CV_CHANNELS, rbm.k, rbm.nh1, rbm.nh2),
             h_cv(get_batch_size(rbm), H_CV_CHANNELS, rbm.nv1, rbm.nv2),
             p_h_a(get_batch_size(rbm), rbm.k, rbm.nh1, rbm.nh2),
             p_h_s(get_batch_size(rbm), rbm.k, rbm.nh1, rbm.nh2),
             w_pos(DYN_W_DIMS),
             w_neg(DYN_W_DIMS),
             v1(get_batch_size(rbm), rbm.nc, rbm.nv1, rbm.nv2),
             vf(get_batch_size(rbm), rbm.nc, rbm.nv1, rbm.nv2),
             h1_a(get_batch_size(rbm), rbm.k, rbm.nh1, rbm.nh2),
             h1_s(get_batch_size(rbm), rbm.k, rbm.nh1, rbm.nh2),
             v2_a(get_batch_size(rbm), rbm.nc, rbm.nv1, rbm.nv2),
             v2_s(get_batch_size(rbm), rbm.nc, rbm.nv1, rbm.nv2),
             h2_a(get_batch_size(rbm), rbm.k, rbm.nh1, rbm.nh2),
             h2_s(get_batch_size(rbm), rbm.k, rbm.nh1, rbm.nh2),
             pool(etl::threads) {
        //Nothign else to init
    }

    void update(RBM& rbm) {
        update_convolutional(rbm, *this);
    }

    template <typename T>
    void train_batch(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context) {
        train_convolutional<Persistent, Denoising, N>(input_batch, expected_batch, context, rbm, *this);
    }

    static std::string name() {
        return std::string("") + (Persistent ? "Persistent " : "") + (Denoising ? "Denoising " : "") + "Contrastive Divergence (dynamic convolutional)";
    }
};

/*!
 * \brief Contrastive Divergence Trainer for RBM.
 */
template <std::size_t N, typename RBM, bool Denoising, typename Enable = void>
using cd_trainer                                                       = base_cd_trainer<N, RBM, false, Denoising>;

/*!
 * \brief Persistent Contrastive Divergence Trainer for RBM.
 */
template <std::size_t N, typename RBM, bool Denoising, typename Enable = void>
using persistent_cd_trainer                                            = base_cd_trainer<N, RBM, true, Denoising>;

/*!
 * \brief CD-1 trainer for RBM
 */
template <typename RBM, bool Denoising>
using cd1_trainer_t = cd_trainer<1, RBM, Denoising>;

/*!
 * \brief PCD-1 trainer for RBM
 */
template <typename RBM, bool Denoising>
using pcd1_trainer_t = persistent_cd_trainer<1, RBM, Denoising>;

} //end of dll namespace
