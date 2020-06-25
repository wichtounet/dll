//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

#define STATIC_IF_DECAY(d, ...) if constexpr (decay == d) { __VA_ARGS__; }

/*!
 * \brief Base class for all standard trainer
 */
template <typename RBM>
struct base_trainer {
    using rmb_t = RBM; ///< The RBM type being trained

    bool init = true; ///< Helper to indicate if first epoch of CD

    /*!
     * \brief Update the gradients given some type of decay
     * \param grad The gradients to update
     * \param rbm The current RBM
     * \param penalty The penalty to apply
     * \tparam decay The type of decay to apply
     */
    template <decay_type decay, typename V, typename G>
    void update_grad(G& grad, const V& value, const RBM& rbm, double penalty) {
        STATIC_IF_DECAY(decay_type::NONE, grad = grad - penalty);
        STATIC_IF_DECAY(decay_type::L1, grad = grad - rbm.l1_weight_cost * abs(value) - penalty);
        STATIC_IF_DECAY(decay_type::L2, grad = grad - rbm.l2_weight_cost * value - penalty);
        STATIC_IF_DECAY(decay_type::L1L2, grad = grad - rbm.l1_weight_cost * abs(value) - rbm.l2_weight_cost * value - penalty);
    }
};

template<typename Grad, typename T>
void apply_clip_gradients(Grad& grad, T t, size_t n){
    auto grad_l2_norm = std::sqrt(etl::sum(grad >> grad) / (n * n));

    if(grad_l2_norm > t){
        grad = grad >> (t / grad_l2_norm);
    }
}

/* The update weights procedure */

/*!
 * \brief Given the gradients of the RBM, update it according to the training
 * configuration.
 */
template <typename RBM, typename Trainer>
void update_normal(RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:update:normal");

    using rbm_t  = RBM;                    ///< The type of the RBM being trained

    //Penalty to be applied to weights and hidden biases
    typename rbm_t::weight w_penalty = 0.0;
    typename rbm_t::weight h_penalty = 0.0;
    typename rbm_t::weight v_penalty = 0.0;

    //Global sparsity method
    if constexpr (rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::GLOBAL_TARGET) {
        auto decay_rate = rbm.decay_rate;
        auto p          = rbm.sparsity_target;
        auto cost       = rbm.sparsity_cost;

        t.q_global_t = decay_rate * t.q_global_t + (1.0 - decay_rate) * t.q_global_batch;

        w_penalty = h_penalty = cost * (t.q_global_t - p);
    }

    //Apply L1/L2 regularization and penalties to the biases

    t.template update_grad<w_decay(rbm_layer_traits<rbm_t>::decay())>(t.w_grad, rbm.w, rbm, w_penalty);
    t.template update_grad<b_decay(rbm_layer_traits<rbm_t>::decay())>(t.b_grad, rbm.b, rbm, h_penalty);
    t.template update_grad<b_decay(rbm_layer_traits<rbm_t>::decay())>(t.c_grad, rbm.c, rbm, v_penalty);

    //Local sparsity method
    if constexpr (rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET) {
        auto decay_rate = rbm.decay_rate;
        auto p          = rbm.sparsity_target;
        auto cost       = rbm.sparsity_cost;

        t.q_local_t = decay_rate * t.q_local_t + (1.0 - decay_rate) * t.q_local_batch;

        auto q_local_penalty = -1 * cost * (t.q_local_t - p);

        t.b_grad += q_local_penalty;
        t.w_grad = bias_add_2d(t.w_grad, q_local_penalty);
    }

    //TODO the batch is not necessary full!
    const auto n_samples = double(etl::dim<0>(t.v1));

    // Gradients clipping
    if constexpr (rbm_layer_traits<rbm_t>::has_clip_gradients()){
        auto grad_t = rbm.gradient_clip;

        apply_clip_gradients(t.w_grad, grad_t, n_samples);
        apply_clip_gradients(t.b_grad, grad_t, n_samples);
        apply_clip_gradients(t.c_grad, grad_t, n_samples);
    }

    // Scale the learning rate with the size of the batch
    auto eps = rbm.learning_rate / n_samples;

    //Apply momentum and learning rate
    if constexpr (rbm_layer_traits<rbm_t>::has_momentum()) {
        auto momentum = rbm.momentum;

        t.w_inc = momentum * t.w_inc + eps * t.w_grad;
        t.b_inc = momentum * t.b_inc + eps * t.b_grad;
        t.c_inc = momentum * t.c_inc + eps * t.c_grad;

        rbm.w += t.w_inc;
        rbm.b += t.b_inc;
        rbm.c += t.c_inc;
    }
    //Apply the learning rate
    else {
        rbm.w += eps * t.w_grad;
        rbm.b += eps * t.b_grad;
        rbm.c += eps * t.c_grad;
    }

    //Check for NaN
    nan_check_deep_3(rbm.w, rbm.b, rbm.c);
}

/*!
 * \brief Given the gradients of the RBM, update it according to the training
 * configuration.
 */
template <typename RBM, typename Trainer>
void update_convolutional(RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:update:conv");

    using rbm_t  = RBM;                    ///< The type of the RBM being trained
    using weight = typename rbm_t::weight; ///< The data type for this layer

    //Penalty to be applied to weights and hidden biases
    weight w_penalty = 0.0;
    weight h_penalty = 0.0;
    weight v_penalty = 0.0;

    //Global sparsity method
    if constexpr (rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::GLOBAL_TARGET) {
        auto decay_rate = rbm.decay_rate;
        auto p          = rbm.sparsity_target;
        auto cost       = rbm.sparsity_cost;

        t.q_global_t = decay_rate * t.q_global_t + (1.0 - decay_rate) * t.q_global_batch;

        w_penalty = h_penalty = cost * (t.q_global_t - p);
    }

    //Apply L1/L2 regularization and penalties to the biases

    t.template update_grad<w_decay(rbm_layer_traits<rbm_t>::decay())>(t.w_grad, rbm.w, rbm, w_penalty);
    t.template update_grad<b_decay(rbm_layer_traits<rbm_t>::decay())>(t.b_grad, rbm.b, rbm, h_penalty);
    t.template update_grad<b_decay(rbm_layer_traits<rbm_t>::decay())>(t.c_grad, rbm.c, rbm, v_penalty);

    //Local sparsity method
    if constexpr (rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET) {
        auto decay_rate = rbm.decay_rate;
        auto p          = rbm.sparsity_target;
        auto cost       = rbm.sparsity_cost;

        t.q_local_t = decay_rate * t.q_local_t + (1.0 - decay_rate) * t.q_local_batch;

        auto q_local_penalty = cost * (t.q_local_t - p);

        t.b_grad -= sum_r(q_local_penalty);

        const auto K   = get_k(rbm);

        auto k_penalty = sum_r(q_local_penalty);
        for (size_t k = 0; k < K; ++k) {
            t.w_grad(k) = t.w_grad(k) - k_penalty(k);
        }
    }

    //Honglak Lee's sparsity method
    if constexpr (rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE) {
        t.w_grad -= rbm.pbias_lambda * t.w_bias;
        t.b_grad -= rbm.pbias_lambda * t.b_bias;
        t.c_grad -= rbm.pbias_lambda * t.c_bias;
    }

    constexpr auto n_samples = RBM::batch_size;
    auto eps             = rbm.learning_rate / n_samples;

    //Apply momentum and learning rate
    if constexpr (rbm_layer_traits<rbm_t>::has_momentum()) {
        auto momentum = rbm.momentum;

        t.w_inc = momentum * t.w_inc + eps * t.w_grad;
        t.b_inc = momentum * t.b_inc + eps * t.b_grad;
        t.c_inc = momentum * t.c_inc + eps * t.c_grad;

        rbm.w += t.w_inc;
        rbm.b += t.b_inc;
        rbm.c += t.c_inc;
    }
    //Apply learning rate only
    else {
        rbm.w += eps * t.w_grad;
        rbm.b += eps * t.b_grad;
        rbm.c += eps * t.c_grad;
    }

    //Check for NaN
    nan_check_deep(rbm.w);
    nan_check_deep(rbm.b);
    nan_check_deep(rbm.c);
}

/* The training procedures */

/*!
 * \brief Compute the gradients for a fully-connected RBM
 */
template <bool Persistent, size_t K, typename InputBatch, typename ExpectedBatch, typename RBM, typename Trainer>
void compute_gradients_normal(InputBatch& input_batch, ExpectedBatch& expected_batch, RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:gradients:normal:batch");

    cpp_assert(etl::dim<0>(input_batch) == etl::dim<0>(expected_batch), "Invalid batch sizes");
    cpp_assert(etl::dim<0>(t.v1) >= etl::dim<0>(input_batch), "Invalid batch sizes");
    cpp_assert(etl::dim<0>(t.vf) >= etl::dim<0>(expected_batch), "Invalid batch sizes");

    cpp_assert(etl::size(input_batch) == etl::size(expected_batch), "Invalid input to compute_gradients_normal");
    cpp_assert(etl::size(t.v1) >= etl::size(input_batch), "Invalid input to compute_gradients_normal");
    cpp_assert(etl::size(t.vf) >= etl::size(expected_batch), "Invalid input to compute_gradients_normal");

    const size_t IB         = etl::dim<0>(input_batch);
    const bool   full_batch = (IB == RBM::batch_size);

    //Copy input/expected for computations
    if(cpp_likely(full_batch)){
        t.v1 = input_batch;
        t.vf = expected_batch;
    } else {
        t.v1 = 0;
        t.vf = 0;

        etl::slice(t.v1, 0, IB) = input_batch;
        etl::slice(t.vf, 0, IB) = expected_batch;
    }

    //First step
    rbm.template batch_activate_hidden<true, true>(t.h1_a, t.h1_s, t.v1, t.v1);

    if (Persistent && t.init) {
        t.p_h_a = t.h1_a;
        t.p_h_s = t.h1_s;
    }

    //CD-1
    if constexpr (Persistent) {
        rbm.template batch_activate_visible<true, false>(t.p_h_a, t.p_h_s, t.v2_a, t.v2_s);
        rbm.template batch_activate_hidden<true, true>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
    } else {
        rbm.template batch_activate_visible<true, false>(t.h1_a, t.h1_s, t.v2_a, t.v2_s);
        rbm.template batch_activate_hidden<true, (K > 1)>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
    }

    //CD-k
    for (size_t k = 1; k < K; ++k) {
        rbm.template batch_activate_visible<true, false>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
        rbm.template batch_activate_hidden<true, true>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
    }

    //Compute the gradients

    {
        dll::auto_timer timer("cd:batch_compute_gradients:std");

        t.w_grad = batch_outer(t.vf, t.h1_a) - batch_outer(t.v2_a, t.h2_a);

        t.b_grad = bias_batch_sum_2d(t.h1_a - t.h2_a);
        t.c_grad = bias_batch_sum_2d(t.vf - t.v2_a);
    }
}

/*!
 * \brief Train a fully-connected RBM.
 */
template <bool Persistent, size_t K, typename InputBatch, typename ExpectedBatch, typename RBM, typename Trainer>
void train_normal(InputBatch& input_batch, ExpectedBatch& expected_batch, rbm_training_context& context, RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:train:normal");

    using namespace etl;

    using rbm_t  = RBM;                    ///< The type of the RBM being trained

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

    if constexpr (rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET) {
        t.q_local_batch = bias_batch_mean_2d(t.h2_a);
    }

    context.batch_sparsity = t.q_global_batch;

    //Update the weights and biases based on the gradients
    t.update(rbm);
}

/*!
 * \brief Compute the gradients for a Convolutional RBM
 */
template <bool Persistent, size_t N, typename Trainer, typename InputBatch, typename ExpectedBatch, typename RBM>
void compute_gradients_conv(InputBatch& input_batch, ExpectedBatch& expected_batch, RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:gradients:conv:batch");

    cpp_assert(etl::dim<0>(input_batch) == etl::dim<0>(expected_batch), "Invalid batch sizes");

    const size_t B        = etl::dim<0>(input_batch);
    const bool full_batch = etl::dim<0>(input_batch) == RBM::batch_size;

    //Copy input/expected for computations
    if(cpp_likely(full_batch)){
        t.v1 = input_batch;
        t.vf = expected_batch;
    } else {
        t.v1 = 0;
        t.vf = 0;

        etl::slice(t.v1, 0, B) = input_batch;
        etl::slice(t.vf, 0, B) = expected_batch;
    }

    //First step
    rbm.template batch_activate_hidden<true, true>(t.h1_a, t.h1_s, t.v1, t.v1);

    if (Persistent && t.init) {
        t.p_h_a = t.h1_a;
        t.p_h_s = t.h1_s;
    }

    //CD-1
    if (Persistent) {
        rbm.template batch_activate_visible<true, false>(t.p_h_a, t.p_h_s, t.v2_a, t.v2_s);
        rbm.template batch_activate_hidden<true, true>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
    } else {
        rbm.template batch_activate_visible<true, false>(t.h1_a, t.h1_s, t.v2_a, t.v2_s);
        rbm.template batch_activate_hidden<true, (N > 1)>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
    }

    //CD-k
    for (size_t k = 1; k < N; ++k) {
        rbm.template batch_activate_visible<true, false>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
        rbm.template batch_activate_hidden<true, true>(t.h2_a, t.h2_s, t.v2_a, t.v2_s);
    }

    //Compute gradients

    {
        dll::auto_timer timer("cd:batch_compute_gradients_conv");

        t.w_pos = etl::ml::convolution_backward_filter(t.vf, t.h1_a);
        t.w_neg = etl::ml::convolution_backward_filter(t.v2_a, t.h2_a);
    }
}

/*!
 * \brief Train a convolutional RBM
 */
template <bool Persistent, size_t N, typename Trainer, typename InputBatch, typename ExpectedBatch, typename RBM>
void train_convolutional(InputBatch& input_batch, ExpectedBatch& expected_batch, rbm_training_context& context, RBM& rbm, Trainer& t) {
    dll::auto_timer timer("cd:train:conv");

    using rbm_t = RBM; ///< The type of the RBM being trained

    compute_gradients_conv<Persistent, N>(input_batch, expected_batch, rbm, t);

    if (Persistent) {
        t.p_h_a = t.h2_a;
        t.p_h_s = t.h2_s;

        t.init = false;
    }

    //Compute the gradients
    t.w_grad = t.w_pos - t.w_neg;
    t.b_grad = bias_batch_mean_4d(t.h1_a - t.h2_a);
    t.c_grad = bias_batch_mean_4d(t.vf - t.v2_a);

    nan_check_deep(t.w_grad);
    nan_check_deep(t.b_grad);
    nan_check_deep(t.c_grad);

    //Compute the mean activation probabilities
    t.q_global_batch = mean(t.h2_a);

    if constexpr (rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET) {
        t.q_local_batch = bias_batch_mean_2d(t.h2_a);
    }

    //Compute the biases for sparsity

    //Only b_bias are supported for now
    if constexpr (rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE && rbm_layer_traits<rbm_t>::bias_mode() == bias_mode::SIMPLE) {
        t.b_bias = mean_r(mean_l(t.h2_a)) - rbm.pbias;
    }

    //Accumulate the sparsity
    context.batch_sparsity = t.q_global_batch;

    //Accumulate the error
    context.batch_error = mean(etl::scale((t.vf - t.v2_a), (t.vf - t.v2_a)));

    //Update the weights and biases based on the gradients
    t.update(rbm);
}

/* The specialized trainers */

/*!
 * \brief Base class for all Contrastive Divergence Trainer.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template <size_t N, typename RBM, bool Persistent, typename Enable = void>
struct base_cd_trainer : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    using rbm_t  = RBM;                    ///< The type of RBM being trained
    using weight = typename rbm_t::weight; ///< The data type for this layer

    static constexpr auto num_hidden  = rbm_t::num_hidden;  ///< The number of hidden units
    static constexpr auto num_visible = rbm_t::num_visible; ///< The number of visible units
    static constexpr auto batch_size  = rbm_t::batch_size;  ///< The batch size of the RBM

    rbm_t& rbm; ///< The RBM being trained

    etl::fast_matrix<weight, batch_size, num_visible> v1; ///< The Input
    etl::fast_matrix<weight, batch_size, num_visible> vf; ///< The Expected Output

    etl::fast_matrix<weight, batch_size, num_hidden> h1_a; ///< The hidden activation probabilites at step one
    etl::fast_matrix<weight, batch_size, num_hidden> h1_s; ///< The hidden states at step one

    etl::fast_matrix<weight, batch_size, num_visible> v2_a; ///< The visible activation probabilites at step N
    etl::fast_matrix<weight, batch_size, num_visible> v2_s; ///< The visible states at step N

    etl::fast_matrix<weight, batch_size, num_hidden> h2_a; ///< The hidden activation probabilites at step N
    etl::fast_matrix<weight, batch_size, num_hidden> h2_s; ///< The hidden states at step N

    //Gradients
    etl::fast_matrix<weight, num_visible, num_hidden> w_grad; ///< The gradients of the weights
    etl::fast_vector<weight, num_hidden> b_grad;              ///< The gradients of the hidden biases
    etl::fast_vector<weight, num_visible> c_grad;             ///< The gradients of the visible biases

    //{{{ Momentum

    etl::fast_matrix<weight, num_visible, num_hidden> w_inc; ///< The gradients of the weights at the previous step for momentum
    etl::fast_vector<weight, num_hidden> b_inc;              ///< The gradients of the hidden biases at the previous step for momentum
    etl::fast_vector<weight, num_visible> c_inc;             ///< The gradients of the visible biases at the previous step for momentum

    //}}} Momentum end

    //{{{ Sparsity

    weight q_global_batch; ///< The global sparsity of the batch
    weight q_global_t; ///< The global sparsity penalty

    etl::fast_matrix<weight, num_hidden> q_local_batch; ///< The local sparsity on the batch
    etl::fast_vector<weight, num_hidden> q_local_t;     ///< The local sparsity penalty

    //}}} Sparsity end

    etl::fast_matrix<weight, batch_size, rbm_t::num_hidden> p_h_a; ///< Beginning of the contrastive divergence chain (activations)
    etl::fast_matrix<weight, batch_size, rbm_t::num_hidden> p_h_s; ///< Beginning of the contrastive divergence chain (samples)

    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm), q_global_t(0.0), q_local_t(0.0) {
        if constexpr (rbm_layer_traits<rbm_t>::has_momentum()) {
            w_inc = 0;
            b_inc = 0;
            c_inc = 0;
        }
    }

    /*!
     * \brief Update the given RBM
     */
    void update(RBM& rbm) {
        update_normal(rbm, *this);
    }

    /*!
     * \brief Train the RBM with one batch of data
     */
    template <typename InputBatch, typename ExpectedBatch>
    void train_batch(InputBatch& input_batch, ExpectedBatch& expected_batch, rbm_training_context& context) {
        train_normal<Persistent, N>(input_batch, expected_batch, context, rbm, *this);
    }

    /*!
     * \brief The name of the trainer
     */
    static std::string name() {
        return std::string("") + (Persistent ? "Persistent " : "") + "Contrastive Divergence";
    }
};

/*!
 * \brief Base class for all Contrastive Divergence Trainer.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template <size_t N, typename RBM, bool Persistent>
struct base_cd_trainer<N, RBM, Persistent, std::enable_if_t<layer_traits<RBM>::is_dynamic() && !layer_traits<RBM>::is_convolutional_rbm_layer()>> : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    using rbm_t  = RBM;                    ///< The type of RBM being trained
    using weight = typename rbm_t::weight; ///< The weight data type

    static constexpr auto batch_size  = rbm_t::batch_size;  ///< The batch size of the RBM

    rbm_t& rbm; ///< The RBM being trained

    etl::dyn_matrix<weight> v1; ///< Input
    etl::dyn_matrix<weight> vf; ///< Expected

    etl::dyn_matrix<weight> h1_a; ///< The hidden activations at step 1
    etl::dyn_matrix<weight> h1_s; ///< The hidden samples at step 1

    etl::dyn_matrix<weight> v2_a; ///< The reconstructed activations at step 1
    etl::dyn_matrix<weight> v2_s; ///< The reconstructed samples at step 1

    etl::dyn_matrix<weight> h2_a; ///< The hidden activations at step K
    etl::dyn_matrix<weight> h2_s; ///< The hidden samples at step K

    //Gradients
    etl::dyn_matrix<weight> w_grad; ///< The gradients of the weights
    etl::dyn_vector<weight> b_grad; ///< The gradients of the hidden biases
    etl::dyn_vector<weight> c_grad; ///< The gradients of the visible biases

    //{{{ Momentum

    etl::dyn_matrix<weight> w_inc; ///< The gradients of the weights at the previous step, for momentum
    etl::dyn_vector<weight> b_inc; ///< The gradients of the hidden biases at the previous step, for momentum
    etl::dyn_vector<weight> c_inc; ///< The gradients of the visible biases at the previous step, for momentum

    //}}} Momentum end

    //{{{ Sparsity

    weight q_global_batch; ///< The global sparsity on the batch
    weight q_global_t;     ///< The global sparsity penalty

    etl::dyn_vector<weight> q_local_batch; ///< The local sparsity on the batch
    etl::dyn_vector<weight> q_local_t;     ///< The local sparsity penalty

    //}}} Sparsity end

    etl::dyn_matrix<weight> p_h_a; ///< Beginning of the contrastive divergence chain (activations)
    etl::dyn_matrix<weight> p_h_s; ///< Beginning of the contrastive divergence chain (samples)

    template <bool M = rbm_layer_traits<rbm_t>::has_momentum(), cpp_disable_iff(M)>
    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm),
              v1(batch_size, rbm.num_visible),
              vf(batch_size, rbm.num_visible),
              h1_a(batch_size, rbm.num_hidden),
              h1_s(batch_size, rbm.num_hidden),
              v2_a(batch_size, rbm.num_visible),
              v2_s(batch_size, rbm.num_visible),
              h2_a(batch_size, rbm.num_hidden),
              h2_s(batch_size, rbm.num_hidden),
              w_grad(rbm.num_visible, rbm.num_hidden),
              b_grad(rbm.num_hidden),
              c_grad(rbm.num_visible),
              w_inc(0, 0),
              b_inc(0),
              c_inc(0),
              q_global_t(0.0),
              q_local_batch(rbm.num_hidden),
              q_local_t(rbm.num_hidden, static_cast<weight>(0.0)),
              p_h_a(batch_size, rbm.num_hidden),
              p_h_s(batch_size, rbm.num_hidden){
        static_assert(!rbm_layer_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template <bool M = rbm_layer_traits<rbm_t>::has_momentum(), cpp_enable_iff(M)>
    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm),
              v1(batch_size, rbm.num_visible),
              vf(batch_size, rbm.num_visible),
              h1_a(batch_size, rbm.num_hidden),
              h1_s(batch_size, rbm.num_hidden),
              v2_a(batch_size, rbm.num_visible),
              v2_s(batch_size, rbm.num_visible),
              h2_a(batch_size, rbm.num_hidden),
              h2_s(batch_size, rbm.num_hidden),
              w_grad(rbm.num_visible, rbm.num_hidden),
              b_grad(rbm.num_hidden),
              c_grad(rbm.num_visible),
              w_inc(rbm.num_visible, rbm.num_hidden, static_cast<weight>(0.0)),
              b_inc(rbm.num_hidden, static_cast<weight>(0.0)),
              c_inc(rbm.num_visible, static_cast<weight>(0.0)),
              q_global_t(0.0),
              q_local_batch(rbm.num_hidden),
              q_local_t(rbm.num_hidden, static_cast<weight>(0.0)),
              p_h_a(batch_size, rbm.num_hidden),
              p_h_s(batch_size, rbm.num_hidden){
        static_assert(rbm_layer_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    /*!
     * \brief Update the given RBM
     */
    void update(RBM& rbm) {
        update_normal(rbm, *this);
    }

    /*!
     * \brief Train the RBM with one batch of data
     */
    template <typename InputBatch, typename ExpectedBatch>
    void train_batch(InputBatch& input_batch, ExpectedBatch& expected_batch, rbm_training_context& context) {
        train_normal<Persistent, N>(input_batch, expected_batch, context, rbm, *this);
    }

    /*!
     * \brief Return the name of the trainer
     */
    static std::string name() {
        return std::string("") + (Persistent ? "Persistent " : "") + "Contrastive Divergence (dynamic)";
    }
};

/*!
 * \brief Specialization of base_cd_trainer for Convolutional RBM.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template <size_t N, typename RBM, bool Persistent>
struct base_cd_trainer<N, RBM, Persistent, std::enable_if_t<!layer_traits<RBM>::is_dynamic() && layer_traits<RBM>::is_convolutional_rbm_layer()>> : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    using rbm_t  = RBM;                    ///< The type of the RBM being trained
    using weight = typename rbm_t::weight; ///< The weight data type

    static constexpr auto K   = rbm_t::K;   ///< The number of filters
    static constexpr auto NC  = rbm_t::NC;  ///< The number of channels
    static constexpr auto NV1 = rbm_t::NV1; ///< The first dimension of the input
    static constexpr auto NV2 = rbm_t::NV2; ///< The second dimension of the input
    static constexpr auto NH1 = rbm_t::NH1; ///< The first dimension of the output
    static constexpr auto NH2 = rbm_t::NH2; ///< The second dimension of the output
    static constexpr auto NW1 = rbm_t::NW1; ///< The second dimension of the filter
    static constexpr auto NW2 = rbm_t::NW2; ///< The second dimension of the filter

    static constexpr auto batch_size = rbm_t::batch_size; ///< The batch size of the trained RBM

    rbm_t& rbm; ///< The RBM being trained

#define W_DIMS K, NC, NW1, NW2

    //Gradients
    etl::fast_matrix<weight, W_DIMS> w_grad; ///< Gradients of the weights weights
    etl::fast_vector<weight, K> b_grad;      ///< Gradients of hidden biases
    etl::fast_vector<weight, NC> c_grad;     ///< Gradients of visible biases

    //{{{ Momentum

    etl::fast_matrix<weight, W_DIMS> w_inc; ///< Gradients of the weights weights of the previous step, for momentum
    etl::fast_vector<weight, K> b_inc;      ///< Gradients of the hidden biases of the previous step, for momentum
    etl::fast_vector<weight, NC> c_inc;     ///< Gradients of the visible biases of the previous step, for momentum

    //}}} Momentum end

    //{{{ Sparsity

    weight q_global_batch; ///< The global batch sparsity
    weight q_global_t;     ///< The global batch sparsity penalty

    conditional_fast_matrix_t<rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET, weight, K, NH1, NH2> q_local_batch; ///< The local batch sparsity
    conditional_fast_matrix_t<rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET, weight, K, NH1, NH2> q_local_t;     ///< The local batch sparsity penalty

    //}}} Sparsity end

    //{{{ Sparsity biases

    conditional_fast_matrix_t<rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE, weight, W_DIMS> w_bias; ///< The sparsity of the weights (for LEE sparsity method)
    conditional_fast_matrix_t<rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE, weight, K> b_bias;      ///< The sparsity of the hidden biases (for LEE sparsity method)
    conditional_fast_matrix_t<rbm_layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE, weight, NC> c_bias;     ///< The sparsity of the visible biases (for LEE sparsity method)

    //}}} Sparsity biases end

    conditional_fast_matrix_t<Persistent, weight, batch_size, K, NH1, NH2> p_h_a; ///< Beginning of the contrastive divergence chain (activations)
    conditional_fast_matrix_t<Persistent, weight, batch_size, K, NH1, NH2> p_h_s; ///< Beginning of the contrastive divergence chain (samples)

    etl::fast_matrix<weight, W_DIMS> w_pos; ///< The positive gradients
    etl::fast_matrix<weight, W_DIMS> w_neg; ///< The negative gradients

    etl::fast_matrix<weight, batch_size, NC, NV1, NV2> v1; ///< Input
    etl::fast_matrix<weight, batch_size, NC, NV1, NV2> vf; ///< Expected

    etl::fast_matrix<weight, batch_size, K, NH1, NH2> h1_a; ///< The hidden activation at step 1
    etl::fast_matrix<weight, batch_size, K, NH1, NH2> h1_s; ///< The hidden samples at step 1

    etl::fast_matrix<weight, batch_size, NC, NV1, NV2> v2_a;                 ///< The visible activation at step K
    conditional_fast_matrix_t<false, weight, batch_size, NC, NV1, NV2> v2_s; ///< The visible samples at step K

    etl::fast_matrix<weight, batch_size, K, NH1, NH2> h2_a;                                 ///< The hidden activation at step K
    conditional_fast_matrix_t<(Persistent || N > 1), weight, batch_size, K, NH1, NH2> h2_s; ///< The hidden samples at step K

    base_cd_trainer(rbm_t& rbm)
            : rbm(rbm),
              w_inc(0.0),
              b_inc(0.0),
              c_inc(0.0),
              q_global_t(0.0),
              q_local_t(0.0),
              w_bias(0.0),
              b_bias(0.0),
              c_bias(0.0) {
        if constexpr (rbm_layer_traits<rbm_t>::has_momentum()) {
            w_inc = 0;
            b_inc = 0;
            c_inc = 0;
        }
    }

    /*!
     * \brief Update the given RBM
     */
    void update(RBM& rbm) {
        update_convolutional(rbm, *this);
    }

    /*!
     * \brief Train the RBM with one batch of data
     */
    template <typename InputBatch, typename ExpectedBatch>
    void train_batch(InputBatch& input_batch, ExpectedBatch& expected_batch, rbm_training_context& context) {
        train_convolutional<Persistent, N>(input_batch, expected_batch, context, rbm, *this);
    }

    /*!
     * \brief Return the name of the trainer
     */
    static std::string name() {
        return std::string("") + (Persistent ? "Persistent " : "") + "Contrastive Divergence (convolutional)";
    }
};

/*!
 * \brief Specialization of base_cd_trainer for dynamic Convolutional RBM.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template <size_t N, typename RBM, bool Persistent>
struct base_cd_trainer<N, RBM, Persistent, std::enable_if_t<layer_traits<RBM>::is_dynamic() && layer_traits<RBM>::is_convolutional_rbm_layer()>> : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    using rbm_t  = RBM;                    ///< The type of the RBM being trained
    using weight = typename rbm_t::weight; ///< The data type

    static constexpr auto batch_size = rbm_t::batch_size; ///< The batch size of the trained RBM

    rbm_t& rbm; ///< The RBM being trained

#define DYN_W_DIMS rbm.k, rbm.nc, rbm.nw1, rbm.nw2

    //Gradients
    etl::dyn_matrix<weight, 4> w_grad; ///< Gradients of shared weights
    etl::dyn_matrix<weight, 1> b_grad; ///< Gradients of hidden biases bk
    etl::dyn_matrix<weight, 1> c_grad; ///< Visible gradient

    //{{{ Momentum

    etl::dyn_matrix<weight, 4> w_inc; ///< Gradients of the weights of the previous step, for momentum
    etl::dyn_matrix<weight, 1> b_inc; ///< Gradients of the hidden biases of the previous step, for momentum
    etl::dyn_matrix<weight, 1> c_inc; ///< Gradients of the visible biases of the previous step, for momentum

    //}}} Momentum end

    //{{{ Sparsity

    weight q_global_batch; ///< The global batch sparsity
    weight q_global_t;     ///< The global batch sparsity penalty

    etl::dyn_matrix<weight, 3> q_local_batch; ///< The local batch sparsity
    etl::dyn_matrix<weight, 3> q_local_t;     ///< The local batch penalty

    //}}} Sparsity end

    //{{{ Sparsity biases

    etl::dyn_matrix<weight, 4> w_bias; ///< The sparsity of the weights (for LEE sparsity method)
    etl::dyn_matrix<weight, 1> b_bias; ///< The sparsity of the hidden biases (for LEE sparsity method)
    etl::dyn_matrix<weight, 1> c_bias; ///< The sparsity of the visible biases (for LEE sparsity method)

    //}}} Sparsity biases end

    etl::dyn_matrix<weight, 4> p_h_a; ///< Beginning of the contrastive divergence chain (activations)
    etl::dyn_matrix<weight, 4> p_h_s; ///< Beginning of the contrastive divergence chain (samples)

    etl::dyn_matrix<weight, 4> w_pos; ///< The positive gradients
    etl::dyn_matrix<weight, 4> w_neg; ///< The negative gradients

    etl::dyn_matrix<weight, 4> v1; ///< Input
    etl::dyn_matrix<weight, 4> vf; ///< Expected

    etl::dyn_matrix<weight, 4> h1_a; ///< The hidden activations at the first step
    etl::dyn_matrix<weight, 4> h1_s; ///< The hidden samples at the first step

    etl::dyn_matrix<weight, 4> v2_a; ///< The visible activations at the first step
    etl::dyn_matrix<weight, 4> v2_s; ///< The visible samples at the first step

    etl::dyn_matrix<weight, 4> h2_a; ///< The hidden activations at the last step
    etl::dyn_matrix<weight, 4> h2_s; ///< The hidden samples at the last step

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
             p_h_a(batch_size, rbm.k, rbm.nh1, rbm.nh2),
             p_h_s(batch_size, rbm.k, rbm.nh1, rbm.nh2),
             w_pos(DYN_W_DIMS),
             w_neg(DYN_W_DIMS),
             v1(batch_size, rbm.nc, rbm.nv1, rbm.nv2),
             vf(batch_size, rbm.nc, rbm.nv1, rbm.nv2),
             h1_a(batch_size, rbm.k, rbm.nh1, rbm.nh2),
             h1_s(batch_size, rbm.k, rbm.nh1, rbm.nh2),
             v2_a(batch_size, rbm.nc, rbm.nv1, rbm.nv2),
             v2_s(batch_size, rbm.nc, rbm.nv1, rbm.nv2),
             h2_a(batch_size, rbm.k, rbm.nh1, rbm.nh2),
             h2_s(batch_size, rbm.k, rbm.nh1, rbm.nh2)
             {
        //Nothign else to init
    }

    /*!
     * \brief Update the given RBM
     */
    void update(RBM& rbm) {
        update_convolutional(rbm, *this);
    }

    /*!
     * \brief Train the RBM with one batch of data
     */
    template <typename InputBatch, typename ExpectedBatch>
    void train_batch(InputBatch& input_batch, ExpectedBatch& expected_batch, rbm_training_context& context) {
        train_convolutional<Persistent, N>(input_batch, expected_batch, context, rbm, *this);
    }

    /*!
     * \brief Return the name of the trainer
     */
    static std::string name() {
        return std::string("") + (Persistent ? "Persistent " : "") + "Contrastive Divergence (dynamic convolutional)";
    }
};

/*!
 * \brief Contrastive Divergence Trainer for RBM.
 */
template <size_t N, typename RBM, typename Enable = void>
using cd_trainer = base_cd_trainer<N, RBM, false>;

/*!
 * \brief Persistent Contrastive Divergence Trainer for RBM.
 */
template <size_t N, typename RBM, typename Enable = void>
using persistent_cd_trainer = base_cd_trainer<N, RBM, true>;

/*!
 * \brief CD-1 trainer for RBM
 */
template <typename RBM>
using cd1_trainer_t = cd_trainer<1, RBM>;

/*!
 * \brief PCD-1 trainer for RBM
 */
template <typename RBM>
using pcd1_trainer_t = persistent_cd_trainer<1, RBM>;

} //end of dll namespace
