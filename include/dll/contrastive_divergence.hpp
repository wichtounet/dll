//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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

#ifndef DLL_CONTRASTIVE_DIVERGENCE_HPP
#define DLL_CONTRASTIVE_DIVERGENCE_HPP

#include "cpp_utils/assert.hpp"             //Assertions

#include "etl/etl.hpp"
#include "etl/convolution.hpp"

#include "batch.hpp"
#include "decay_type.hpp"
#include "layer_traits.hpp"
#include "parallel.hpp"

namespace dll {

/*!
 * \brief Base class for all standard trainer
 */
template<typename RBM>
struct base_trainer {
    typedef RBM rbm_t;

    bool init = false;

    template<typename T1, typename T2, bool M = layer_traits<rbm_t>::has_momentum(), cpp::enable_if_u<M> = cpp::detail::dummy>
    T2& get_fgrad(T1& , T2& inc){
        return inc;
    }

    template<typename T1, typename T2, bool M = layer_traits<rbm_t>::has_momentum(), cpp::disable_if_u<M> = cpp::detail::dummy>
    T1& get_fgrad(T1& grad, T2& ){
        return grad;
    }

    template<typename V, typename G>
    void update_grad(G& grad, const V& value, const RBM& rbm, decay_type decay, double penalty){
        if(decay == decay_type::L1){
            grad = grad - rbm.l1_weight_cost * abs(value) - penalty;
        } else if(decay == decay_type::L2){
            grad = grad - rbm.l2_weight_cost * value - penalty;
        } else if(decay == decay_type::L1L2){
            grad = grad - rbm.l1_weight_cost * abs(value) - rbm.l2_weight_cost * value - penalty;
        } else {
            grad = grad - penalty;
        }
    }
};

/* Some utilities */

template<typename RBM, typename C, cpp::enable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
auto reshape_nv1(RBM& rbm, C&& container){
    return etl::reshape(container, rbm.num_visible, 1);
}

template<typename RBM, typename C, cpp::disable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
auto reshape_nv1(RBM&, C&& container){
    return etl::reshape<RBM::num_visible, 1>(container);
}

template<typename RBM, typename C, cpp::enable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
auto reshape_1nh(RBM& rbm, C&& container){
    return etl::reshape(container, 1, rbm.num_hidden);
}

template<typename RBM, typename C, cpp::disable_if_u<layer_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
auto reshape_1nh(RBM&, C&& container){
    return etl::reshape<1, RBM::num_hidden>(container);
}

/* The update weights procedure */

template<typename RBM, typename Trainer>
void update_normal(RBM& rbm, Trainer& t){
    using rbm_t = RBM;

    //Penalty to be applied to weights and hidden biases
    typename rbm_t::weight w_penalty = 0.0;
    typename rbm_t::weight h_penalty = 0.0;
    typename rbm_t::weight v_penalty = 0.0;

    //Global sparsity method
    if(layer_traits<rbm_t>::sparsity_method() == sparsity_method::GLOBAL_TARGET){
        auto decay_rate = rbm.decay_rate;
        auto p = rbm.sparsity_target;
        auto cost = rbm.sparsity_cost;

        t.q_global_t = decay_rate * t.q_global_t + (1.0 - decay_rate) * t.q_global_batch;

        w_penalty = h_penalty = cost * (t.q_global_t - p);
    }

    //Apply L1/L2 regularization and penalties to the biases

    t.update_grad(t.w_grad, rbm.w, rbm, w_decay(layer_traits<rbm_t>::decay()), w_penalty);
    t.update_grad(t.b_grad, rbm.b, rbm, b_decay(layer_traits<rbm_t>::decay()), h_penalty);
    t.update_grad(t.c_grad, rbm.c, rbm, b_decay(layer_traits<rbm_t>::decay()), v_penalty);

    //Local sparsity method
    if(layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
        auto decay_rate = rbm.decay_rate;
        auto p = rbm.sparsity_target;
        auto cost = rbm.sparsity_cost;

        t.q_local_t = decay_rate * t.q_local_t + (1.0 - decay_rate) * t.q_local_batch;

        auto q_local_penalty = cost * (t.q_local_t - p);

        t.b_grad -= q_local_penalty;

        for(std::size_t i = 0; i < num_hidden(rbm); ++i){
            for(std::size_t j = 0; j < num_visible(rbm); ++j){
                t.w_grad(j, i) -= q_local_penalty(i);
            }
        }
    }

    //Apply momentum and learning rate
    if(layer_traits<rbm_t>::has_momentum()){
        auto momentum = rbm.momentum;
        auto eps = rbm.learning_rate;

        t.w_inc = momentum * t.w_inc + eps * t.w_grad;
        t.b_inc = momentum * t.b_inc + eps * t.b_grad;
        t.c_inc = momentum * t.c_inc + eps * t.c_grad;
    }
    //Apply the learning rate
    else {
        auto eps = rbm.learning_rate;

        t.w_grad *= eps;
        t.b_grad *= eps;
        t.c_grad *= eps;
    }

    //Update the weights and biases
    //with the final gradients (if not momentum, these are the real gradients)
    rbm.w += t.get_fgrad(t.w_grad, t.w_inc);
    rbm.b += t.get_fgrad(t.b_grad, t.b_inc);
    rbm.c += t.get_fgrad(t.c_grad, t.c_inc);

    //Check for NaN
    nan_check_deep_3(rbm.w, rbm.b, rbm.c);
}

template<typename RBM, typename Trainer>
void update_convolutional(RBM& rbm, Trainer& t){
    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    constexpr const auto NC = rbm_t::NC;
    constexpr const auto NW1 = rbm_t::NW1;
    constexpr const auto NW2 = rbm_t::NW2;

    //Penalty to be applied to weights and hidden biases
    weight w_penalty = 0.0;
    weight h_penalty = 0.0;
    weight v_penalty = 0.0;

    //Global sparsity method
    if(layer_traits<rbm_t>::sparsity_method() == sparsity_method::GLOBAL_TARGET){
        auto decay_rate = rbm.decay_rate;
        auto p = rbm.sparsity_target;
        auto cost = rbm.sparsity_cost;

        t.q_global_t = decay_rate * t.q_global_t + (1.0 - decay_rate) * t.q_global_batch;

        w_penalty = h_penalty = cost * (t.q_global_t - p);
    }

    //Apply L1/L2 regularization and penalties to the biases

    t.update_grad(t.w_grad, rbm.w, rbm, w_decay(layer_traits<rbm_t>::decay()), w_penalty);
    t.update_grad(t.b_grad, rbm.b, rbm, b_decay(layer_traits<rbm_t>::decay()), h_penalty);
    t.update_grad(t.c_grad, rbm.c, rbm, b_decay(layer_traits<rbm_t>::decay()), v_penalty);

    //Local sparsity method
    if(layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
        auto decay_rate = rbm.decay_rate;
        auto p = rbm.sparsity_target;
        auto cost = rbm.sparsity_cost;

        t.q_local_t = decay_rate * t.q_local_t + (1.0 - decay_rate) * t.q_local_batch;

        auto q_local_penalty = cost * (t.q_local_t - p);

        t.b_grad -= sum_r(q_local_penalty);

        auto k_penalty = etl::rep<NW1, NW2>(sum_r(q_local_penalty));
        for(std::size_t channel = 0; channel < NC; ++channel){
            t.w_grad(channel) = t.w_grad(channel) - k_penalty;
        }
    }

    //Honglak Lee's sparsity method
    if(layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE){
        auto eps = rbm.learning_rate;

        t.w_grad -= rbm.pbias_lambda * (1.0 / eps) * t.w_bias;
        t.b_grad -= rbm.pbias_lambda * (1.0 / eps) * t.b_bias;
        t.c_grad -= rbm.pbias_lambda * (1.0 / eps) * t.c_bias;
    }

    //Apply momentum and learning rate
    if(layer_traits<rbm_t>::has_momentum()){
        auto momentum = rbm.momentum;
        auto eps = rbm.learning_rate;

        t.w_inc = momentum * t.w_inc + eps * t.w_grad;
        t.b_inc = momentum * t.b_inc + eps * t.b_grad;
        t.c_inc = momentum * t.c_inc + eps * t.c_grad;
    }
    //Apply learning rate only
    else {
        auto eps = rbm.learning_rate;

        t.w_grad *= eps;
        t.b_grad *= eps;
        t.c_grad *= eps;
    }

    //Update the weights and biases with the final gradients (if not momentum,
    //these are the real gradients)
    rbm.w += t.get_fgrad(t.w_grad, t.w_inc);
    rbm.b += t.get_fgrad(t.b_grad, t.b_inc);
    rbm.c += t.get_fgrad(t.c_grad, t.c_inc);

    //Check for NaN
    nan_check_deep(rbm.w);
    nan_check_deep(rbm.b);
    nan_check_deep(rbm.c);
}

/* The training procedures */

template<bool Persistent, std::size_t K, typename T, typename RBM, typename Trainer>
void train_normal(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context, RBM& rbm, Trainer& t){
    cpp_assert(input_batch.size() > 0, "Invalid batch size");
    cpp_assert(input_batch.size() <= get_batch_size(rbm), "Invalid batch size");
    cpp_assert(input_batch.begin()->size() == input_size(rbm), "The size of the training sample must match visible units");

    using namespace etl;
    using rbm_t = RBM;

    maybe_parallel_foreach_pair_i(t.pool, input_batch.begin(), input_batch.end(), expected_batch.begin(), expected_batch.end(),
            [&](const auto& input, const auto& expected, std::size_t i)
    {
        //Copy input/expected for computations
        t.v1(i) = input;
        t.vf(i) = expected;

        //First step
        rbm.template activate_hidden<true, true>(t.h1_a(i), t.h1_s(i), t.v1(i), t.v1(i), t.ht(i));

        if(Persistent && t.init){
            t.p_h_a(i) = t.h1_a(i);
            t.p_h_s(i) = t.h1_s(i);
        }

        //CD-1
        if(Persistent){
            rbm.template activate_visible<true, false>(t.p_h_a(i), t.p_h_s(i), t.v2_a(i), t.v2_s(i), t.vt(i));
            rbm.template activate_hidden<true, true>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.ht(i));
        } else {
            rbm.template activate_visible<true, false>(t.h1_a(i), t.h1_s(i), t.v2_a(i), t.v2_s(i), t.vt(i));
            rbm.template activate_hidden<true, (K > 1)>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.ht(i));
        }

        //CD-k
        for(std::size_t k = 1; k < K; ++k){
            rbm.template activate_visible<true, false>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.vt(i));
            rbm.template activate_hidden<true, true>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.ht(i));
        }

        //The following lines are equivalent to mmul(vf, h1_a) - mmul(v2_a, h2_a)
        //Doing them this way is significantly faster than computing the two matrix mutplications
        //and doing the subtraction later

        auto g = t.w_grad_b(i);

        //Reset the batch gradients
        g = 0;

        auto a1 = reshape_nv1(rbm, t.vf(i));
        auto b1 = reshape_1nh(rbm, t.h1_a(i));
        auto a2 = reshape_nv1(rbm, t.v2_a(i));
        auto b2 = reshape_1nh(rbm, t.h2_a(i));

        for(std::size_t i2 = 0; i2 < rows(a1); i2++){
            for(std::size_t k = 0; k < columns(a1); k++){
                for(std::size_t j = 0; j < columns(b1); j++){
                    g(i2,j) += a1(i2,k) * b1(k,j) - a2(i2,k) * b2(k,j);
                }
            }
        }
    });

    if(Persistent){
        t.p_h_a = t.h2_a;
        t.p_h_s = t.h2_s;

        t.init = false;
    }

    context.reconstruction_error += mean((t.vf - t.v2_a) * (t.vf - t.v2_a));

    //Compute the gradients
    t.w_grad = mean_l(t.w_grad_b);
    t.b_grad = mean_l(t.h1_a - t.h2_a);
    t.c_grad = mean_l(t.vf - t.v2_a);

    nan_check_deep_3(t.w_grad, t.b_grad, t.c_grad);

    //Compute the mean activation probabilities
    t.q_global_batch = mean(t.h2_a);

    if(layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
        t.q_local_batch = mean_l(t.h2_a);
    }

    //Accumulate the sparsity
    context.sparsity += t.q_global_batch;

    //Update the weights and biases based on the gradients
    t.update(rbm);
}

template<bool Persistent, bool Denoising, std::size_t N, typename Trainer, typename T, typename RBM>
void train_convolutional(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context, RBM& rbm, Trainer& t){
    cpp_assert(input_batch.size() > 0, "Invalid batch size");
    cpp_assert(input_batch.size() <= get_batch_size(rbm), "Invalid batch size");
    cpp_assert(input_batch.size() == expected_batch.size(), "Batches do not match");
    cpp_assert(input_batch.begin()->size() == input_size(rbm), "The size of the training sample must match visible units");

    using rbm_t = RBM;

    maybe_parallel_foreach_pair_i(t.pool, input_batch.begin(), input_batch.end(), expected_batch.begin(), expected_batch.end(),
            [&](const auto& input, const auto& expected, std::size_t i)
    {
        constexpr const auto K = rbm_t::K;
        constexpr const auto NC = rbm_t::NC;

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
            rbm.template activate_hidden<true, (K > 1)>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.v_cv(i));
        }

        //CD-k
        for(std::size_t k = 1; k < N; ++k){
            rbm.template activate_visible<true, false>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.h_cv(i));
            rbm.template activate_hidden<true, true>(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.v_cv(i));
        }

        //Compute gradients

        for(std::size_t channel = 0; channel < NC; ++channel){
            for(std::size_t k = 0; k < K; ++k){
                if(Denoising){
                    etl::convolve_2d_valid(t.vf(i)(channel), fflip(t.h1_a(i)(k)), t.w_pos(i)(channel)(k));
                    etl::convolve_2d_valid(t.v2_a(i)(channel), fflip(t.h2_a(i)(k)), t.w_neg(i)(channel)(k));
                } else {
                    etl::convolve_2d_valid(t.v1(i)(channel), fflip(t.h1_a(i)(k)), t.w_pos(i)(channel)(k));
                    etl::convolve_2d_valid(t.v2_a(i)(channel), fflip(t.h2_a(i)(k)), t.w_neg(i)(channel)(k));
                }
            }
        }
    });

    if(Persistent){
        t.p_h_a = t.h2_a;
        t.p_h_s = t.h2_s;

        t.init = false;
    }

    //Compute the gradients
    t.w_grad = mean_l(t.w_pos - t.w_neg);
    t.b_grad = mean_r(mean_l(t.h1_a - t.h2_a));

    if(Denoising){
        t.c_grad = mean_r(mean_l(t.vf - t.v2_a));
    } else {
        t.c_grad = mean_r(mean_l(t.v1 - t.v2_a));
    }

    nan_check_deep(t.w_grad);
    nan_check_deep(t.b_grad);
    nan_check_deep(t.c_grad);

    //Compute the mean activation probabilities
    t.q_global_batch = mean(t.h2_a);

    if(layer_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
        t.q_local_batch = mean_l(t.h2_a);
    }

    //Compute the biases for sparsity

    if(layer_traits<rbm_t>::bias_mode() == bias_mode::SIMPLE){
        t.b_bias = mean_r(mean_l(t.h2_a)) - rbm.pbias;
    }

    //Accumulate the sparsity
    context.sparsity += t.q_global_batch;

    //Accumulate the error
    if(Denoising){
        context.reconstruction_error += mean((t.vf - t.v2_a) * (t.vf - t.v2_a));
    } else {
        context.reconstruction_error += mean((t.v1 - t.v2_a) * (t.v1 - t.v2_a));
    }

    //Update the weights and biases based on the gradients
    t.update(rbm);
}


/* The specialized trainers */

/*!
 * \brief Base class for all Contrastive Divergence Trainer.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template<std::size_t N, typename RBM, bool Persistent, bool Denoising, typename Enable = void>
struct base_cd_trainer : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    static constexpr const auto num_hidden = rbm_t::num_hidden;
    static constexpr const auto num_visible = rbm_t::num_visible;

    static constexpr const auto batch_size = layer_traits<rbm_t>::batch_size();

    rbm_t& rbm;

    etl::fast_matrix<weight, batch_size, num_visible> v1; //Input
    etl::fast_matrix<weight, batch_size, num_visible> vf; //Expected

    etl::fast_matrix<weight, batch_size, num_hidden> h1_a;
    etl::fast_matrix<weight, batch_size, num_hidden> h1_s;

    etl::fast_matrix<weight, batch_size, num_visible> v2_a;
    etl::fast_matrix<weight, batch_size, num_visible> v2_s;

    etl::fast_matrix<weight, batch_size, num_hidden> h2_a;
    etl::fast_matrix<weight, batch_size, num_hidden> h2_s;

    etl::fast_matrix<weight, batch_size, 1, num_hidden> ht;
    etl::fast_matrix<weight, batch_size, num_visible, 1> vt;

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

    thread_pool<layer_traits<rbm_t>::is_parallel()> pool;

    template<bool M = layer_traits<rbm_t>::has_momentum(), cpp::disable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t& rbm) : rbm(rbm), q_global_t(0.0), q_local_t(0.0) {
        static_assert(!layer_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = layer_traits<rbm_t>::has_momentum(), cpp::enable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t& rbm) : rbm(rbm), w_inc(0.0), b_inc(0.0), c_inc(0.0), q_global_t(0.0), q_local_t(0.0) {
        static_assert(layer_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update(RBM& rbm){
        update_normal(rbm, *this);
    }

    template<typename T>
    void train_batch(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context){
        train_normal<Persistent, N>(input_batch, expected_batch, context, rbm, *this);
    }

    static std::string name(){
        return
                std::string("")
            +   (Persistent ? "Persistent " : "")
            +   (Denoising ? "Denoising " : "")
            +   "Contrastive Divergence";
    }
};

/*!
 * \brief Base class for all Contrastive Divergence Trainer.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template<std::size_t N, typename RBM, bool Persistent, bool Denoising>
struct base_cd_trainer<N, RBM, Persistent, Denoising, std::enable_if_t<layer_traits<RBM>::is_dynamic()>> : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    typedef RBM rbm_t;

    typedef typename rbm_t::weight weight;

    rbm_t& rbm;

    etl::dyn_matrix<weight> v1; //Input
    etl::dyn_matrix<weight> vf; //Expected

    etl::dyn_matrix<weight> h1_a;
    etl::dyn_matrix<weight> h1_s;

    etl::dyn_matrix<weight> v2_a;
    etl::dyn_matrix<weight> v2_s;

    etl::dyn_matrix<weight> h2_a;
    etl::dyn_matrix<weight> h2_s;

    etl::dyn_matrix<weight, 3> ht;
    etl::dyn_matrix<weight, 3> vt;

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

    thread_pool<layer_traits<rbm_t>::is_parallel()> pool;

    template<bool M = layer_traits<rbm_t>::has_momentum(), cpp::disable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t& rbm) : rbm(rbm),
            v1(get_batch_size(rbm), rbm.num_visible),
            vf(get_batch_size(rbm), rbm.num_visible),
            h1_a(get_batch_size(rbm), rbm.num_hidden), h1_s(get_batch_size(rbm), rbm.num_hidden),
            v2_a(get_batch_size(rbm), rbm.num_visible), v2_s(get_batch_size(rbm), rbm.num_visible),
            h2_a(get_batch_size(rbm), rbm.num_hidden), h2_s(get_batch_size(rbm), rbm.num_hidden),
            ht(get_batch_size(rbm), 1UL, rbm.num_hidden), vt(get_batch_size(rbm), rbm.num_visible, 1UL),
            w_grad_b(get_batch_size(rbm), rbm.num_visible, rbm.num_hidden),
            w_grad(rbm.num_visible, rbm.num_hidden), b_grad(rbm.num_hidden), c_grad(rbm.num_visible),
            w_inc(0,0), b_inc(0), c_inc(0),
            q_global_t(0.0),
            q_local_batch(rbm.num_hidden), q_local_t(rbm.num_hidden, static_cast<weight>(0.0)),
            p_h_a(get_batch_size(rbm), rbm.num_hidden), p_h_s(get_batch_size(rbm), rbm.num_hidden)
    {
        static_assert(!layer_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = layer_traits<rbm_t>::has_momentum(), cpp::enable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t& rbm) : rbm(rbm),
            v1(get_batch_size(rbm), rbm.num_visible),
            vf(get_batch_size(rbm), rbm.num_visible),
            h1_a(get_batch_size(rbm), rbm.num_hidden), h1_s(get_batch_size(rbm), rbm.num_hidden),
            v2_a(get_batch_size(rbm), rbm.num_visible), v2_s(get_batch_size(rbm), rbm.num_visible),
            h2_a(get_batch_size(rbm), rbm.num_hidden), h2_s(get_batch_size(rbm), rbm.num_hidden),
            ht(get_batch_size(rbm), 1UL, rbm.num_hidden), vt(get_batch_size(rbm), rbm.num_visible, 1UL),
            w_grad_b(get_batch_size(rbm), rbm.num_visible, rbm.num_hidden),
            w_grad(rbm.num_visible, rbm.num_hidden), b_grad(rbm.num_hidden), c_grad(rbm.num_visible),
            w_inc(rbm.num_visible, rbm.num_hidden, static_cast<weight>(0.0)), b_inc(rbm.num_hidden, static_cast<weight>(0.0)), c_inc(rbm.num_visible, static_cast<weight>(0.0)),
            q_global_t(0.0), q_local_batch(rbm.num_hidden), q_local_t(rbm.num_hidden, static_cast<weight>(0.0)),
            p_h_a(get_batch_size(rbm), rbm.num_hidden), p_h_s(get_batch_size(rbm), rbm.num_hidden)
    {
        static_assert(layer_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update(RBM& rbm){
        update_normal(rbm, *this);
    }

    template<typename T>
    void train_batch(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context){
        train_normal<Persistent, N>(input_batch, expected_batch, context, rbm, *this);
    }

    static std::string name(){
        return
                std::string("")
            +   (Persistent ? "Persistent " : "")
            +   (Denoising ? "Denoising " : "")
            +   "Contrastive Divergence (dynamic)";
    }
};

//This allows to create a fast matrix type with an effective size of zero
//when it is not used, although this fast matrix is viewed as having
//the correct size

template<bool C, typename W, std::size_t... Dims>
struct conditional_fast_matrix {
    using type = std::conditional_t<
        C,
        etl::fast_matrix<W, Dims...>,
        etl::fast_matrix_impl<W, std::array<W, 0>, Dims...>>;
};

template<bool C, typename W, std::size_t... Dims>
using conditional_fast_matrix_t = typename conditional_fast_matrix<C, W, Dims...>::type;

/*!
 * \brief Specialization of base_cd_trainer for Convolutional RBM.
 *
 * This class provides update which applies the gradients to the RBM.
 */
template<std::size_t N, typename RBM, bool Persistent, bool Denoising>
struct base_cd_trainer<N, RBM, Persistent, Denoising, std::enable_if_t<layer_traits<RBM>::is_convolutional()>> : base_trainer<RBM> {
    static_assert(N > 0, "(P)CD-0 is not a valid training method");

    using rbm_t = RBM;

    static constexpr const auto K = rbm_t::K;
    static constexpr const auto NC = rbm_t::NC;
    static constexpr const auto NV1 = rbm_t::NV1;
    static constexpr const auto NV2 = rbm_t::NV2;
    static constexpr const auto NH1 = rbm_t::NH1;
    static constexpr const auto NH2 = rbm_t::NH2;
    static constexpr const auto NW1 = rbm_t::NW1;
    static constexpr const auto NW2 = rbm_t::NW2;

    static constexpr const auto batch_size = layer_traits<rbm_t>::batch_size();

    typedef typename rbm_t::weight weight;

    rbm_t& rbm;

    //Gradients
    etl::fast_matrix<weight, NC, K, NW1, NW2> w_grad;  //Gradients of shared weights
    etl::fast_vector<weight, K> b_grad;              //Gradients of hidden biases bk
    etl::fast_vector<weight, NC> c_grad;             //Visible gradient

    //{{{ Momentum

    etl::fast_matrix<weight, NC, K, NW1, NW2> w_inc;
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

    conditional_fast_matrix_t<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE, weight, NC, K, NW1, NW2> w_bias;
    conditional_fast_matrix_t<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE, weight, K> b_bias;
    conditional_fast_matrix_t<layer_traits<rbm_t>::sparsity_method() == sparsity_method::LEE, weight, NC> c_bias;

    //}}} Sparsity biases end

    etl::fast_matrix<weight, batch_size, 2, K, NH1, NH2> v_cv;
    etl::fast_matrix<weight, batch_size, 2, NV1, NV2> h_cv;

    conditional_fast_matrix_t<Persistent, weight, batch_size, K, NH1, NH2> p_h_a;
    conditional_fast_matrix_t<Persistent, weight, batch_size, K, NH1, NH2> p_h_s;

    etl::fast_matrix<weight, batch_size, NC, K, NW1, NW2> w_pos;
    etl::fast_matrix<weight, batch_size, NC, K, NW1, NW2> w_neg;

    etl::fast_matrix<weight, batch_size, NC, NV1, NV2> v1; //Input
    conditional_fast_matrix_t<Denoising, weight, batch_size, NC, NV1, NV2> vf; //Expected

    etl::fast_matrix<weight, batch_size, K, NH1, NH2> h1_a;
    etl::fast_matrix<weight, batch_size, K, NH1, NH2> h1_s;

    etl::fast_matrix<weight, batch_size, NC, NV1, NV2> v2_a;
    conditional_fast_matrix_t<false, weight, batch_size, NC, NV1, NV2> v2_s;

    etl::fast_matrix<weight, batch_size, K, NH1, NH2> h2_a;
    conditional_fast_matrix_t<(K > 1), weight, batch_size, K, NH1, NH2> h2_s;

    thread_pool<layer_traits<rbm_t>::is_parallel()> pool;

    template<bool M = layer_traits<rbm_t>::has_momentum(), cpp::disable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t& rbm) : rbm(rbm),
            q_global_t(0.0), q_local_t(0.0),
            w_bias(0.0), b_bias(0.0), c_bias(0.0) {
        static_assert(!layer_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = layer_traits<rbm_t>::has_momentum(), cpp::enable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t& rbm) : rbm(rbm),
            w_inc(0.0), b_inc(0.0), c_inc(0.0),
            q_global_t(0.0), q_local_t(0.0),
            w_bias(0.0), b_bias(0.0), c_bias(0.0) {
        static_assert(layer_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update(RBM& rbm){
        update_convolutional(rbm, *this);
    }

    template<typename T>
    void train_batch(const dll::batch<T>& input_batch, const dll::batch<T>& expected_batch, rbm_training_context& context){
        train_convolutional<Persistent, Denoising, N>(input_batch, expected_batch, context, rbm, *this);
    }

    static std::string name(){
        return
                std::string("")
            +   (Persistent ? "Persistent " : "")
            +   (Denoising ? "Denoising " : "")
            +   "Contrastive Divergence (convolutional)";
    }
};

/*!
 * \brief Contrastive Divergence Trainer for RBM.
 */
template<std::size_t N, typename RBM, bool Denoising, typename Enable = void>
using cd_trainer = base_cd_trainer<N, RBM, false, Denoising>;

/*!
 * \brief Persistent Contrastive Divergence Trainer for RBM.
 */
template<std::size_t N, typename RBM, bool Denoising, typename Enable = void>
using persistent_cd_trainer = base_cd_trainer<N, RBM, true, Denoising>;

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

#endif
