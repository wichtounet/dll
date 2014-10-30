//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
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
#include "rbm_traits.hpp"
#include "parallel.hpp"

namespace dll {

/*!
 * \brief Base class for all standard trainer
 */
template<typename RBM>
struct base_trainer {
    typedef RBM rbm_t;

    template<typename T1, typename T2, bool M = rbm_traits<rbm_t>::has_momentum(), cpp::enable_if_u<M> = cpp::detail::dummy>
    T2& get_fgrad(T1& , T2& inc){
        return inc;
    }

    template<typename T1, typename T2, bool M = rbm_traits<rbm_t>::has_momentum(), cpp::disable_if_u<M> = cpp::detail::dummy>
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

template<typename RBM, typename Trainer>
void update_weights_normal(RBM& rbm, Trainer& t){
    using rbm_t = RBM;

    //Penalty to be applied to weights and hidden biases
    typename rbm_t::weight w_penalty = 0.0;
    typename rbm_t::weight h_penalty = 0.0;
    typename rbm_t::weight v_penalty = 0.0;

    //Global sparsity method
    if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::GLOBAL_TARGET){
        auto decay_rate = rbm.decay_rate;
        auto p = rbm.sparsity_target;
        auto cost = rbm.sparsity_cost;

        t.q_global_t = decay_rate * t.q_global_t + (1.0 - decay_rate) * t.q_global_batch;

        w_penalty = h_penalty = cost * (t.q_global_t - p);
    }

    //Apply L1/L2 regularization and penalties to the biases

    t.update_grad(t.w_grad, rbm.w, rbm, w_decay(rbm_traits<rbm_t>::decay()), w_penalty);
    t.update_grad(t.b_grad, rbm.b, rbm, b_decay(rbm_traits<rbm_t>::decay()), h_penalty);
    t.update_grad(t.c_grad, rbm.c, rbm, b_decay(rbm_traits<rbm_t>::decay()), v_penalty);

    //Local sparsity method
    if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
        auto decay_rate = rbm.decay_rate;
        auto p = rbm.sparsity_target;
        auto cost = rbm.sparsity_cost;

        t.q_local_t = decay_rate * t.q_local_t + (1.0 - decay_rate) * t.q_local_batch;

        t.q_local_penalty = cost * (t.q_local_t - p);

        t.b_grad -= t.q_local_penalty;

        for(std::size_t i = 0; i < num_hidden(rbm); ++i){
            for(std::size_t j = 0; j < num_visible(rbm); ++j){
                t.w_grad(j, i) -= t.q_local_penalty(i);
            }
        }
    }

    //Apply momentum and learning rate
    if(rbm_traits<rbm_t>::has_momentum()){
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

/*!
 * \brief Base class for all Contrastive Divergence Trainer.
 *
 * This class provides update_weights which applies the gradients to the RBM.
 */
template<typename RBM, typename Enable = void>
struct base_cd_trainer : base_trainer<RBM> {
    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    static constexpr const auto num_hidden = rbm_t::num_hidden;
    static constexpr const auto num_visible = rbm_t::num_visible;

    static constexpr const auto batch_size = rbm_traits<rbm_t>::batch_size();

    etl::fast_matrix<weight, batch_size, num_visible> v1;

    etl::fast_matrix<weight, batch_size, num_hidden> h1_a;
    etl::fast_matrix<weight, batch_size, num_hidden> h1_s;

    etl::fast_matrix<weight, batch_size, num_visible> v2_a;
    etl::fast_matrix<weight, batch_size, num_visible> v2_s;

    etl::fast_matrix<weight, batch_size, num_hidden> h2_a;
    etl::fast_matrix<weight, batch_size, num_hidden> h2_s;

    etl::fast_matrix<weight, batch_size, 1, num_hidden> ht;
    etl::fast_matrix<weight, batch_size, num_visible, 1> vt;

    //Batch Gradients
    etl::fast_matrix<weight, batch_size, num_visible, num_hidden> w_grad_b;
    etl::fast_matrix<weight, batch_size, num_hidden> b_grad_b;
    etl::fast_matrix<weight, batch_size, num_visible> c_grad_b;

    //Reduced Gradients
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
    etl::fast_vector<weight, num_hidden> q_local_penalty;

    //}}} Sparsity end

    thread_pool pool;

    template<bool M = rbm_traits<rbm_t>::has_momentum(), cpp::disable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t&) : q_global_t(0.0), q_local_t(0.0) {
        static_assert(!rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = rbm_traits<rbm_t>::has_momentum(), cpp::enable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t&) : w_inc(0.0), b_inc(0.0), c_inc(0.0), q_global_t(0.0), q_local_t(0.0) {
        static_assert(rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update_weights(RBM& rbm){
        update_weights_normal(rbm, *this);
    }
};

/*!
 * \brief Base class for all Contrastive Divergence Trainer.
 *
 * This class provides update_weights which applies the gradients to the RBM.
 */
template<typename RBM>
struct base_cd_trainer<RBM, std::enable_if_t<rbm_traits<RBM>::is_dynamic()>> : base_trainer<RBM> {
    typedef RBM rbm_t;

    typedef typename rbm_t::weight weight;

    etl::dyn_matrix<weight> v1;

    etl::dyn_matrix<weight> h1_a;
    etl::dyn_matrix<weight> h1_s;

    etl::dyn_matrix<weight> v2_a;
    etl::dyn_matrix<weight> v2_s;

    etl::dyn_matrix<weight> h2_a;
    etl::dyn_matrix<weight> h2_s;

    etl::dyn_matrix<weight, 3> ht;
    etl::dyn_matrix<weight, 3> vt;

    //Batch Gradients
    etl::dyn_matrix<weight, 3> w_grad_b;
    etl::dyn_matrix<weight> b_grad_b;
    etl::dyn_matrix<weight> c_grad_b;

    //Reduced Gradients
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
    etl::dyn_vector<weight> q_local_penalty;

    //}}} Sparsity end

    thread_pool pool;

    template<bool M = rbm_traits<rbm_t>::has_momentum(), cpp::disable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t& rbm) :
            v1(get_batch_size(rbm), rbm.num_visible),
            h1_a(get_batch_size(rbm), rbm.num_hidden), h1_s(get_batch_size(rbm), rbm.num_hidden),
            v2_a(get_batch_size(rbm), rbm.num_visible), v2_s(get_batch_size(rbm), rbm.num_visible),
            h2_a(get_batch_size(rbm), rbm.num_hidden), h2_s(get_batch_size(rbm), rbm.num_hidden),
            ht(get_batch_size(rbm), 1UL, rbm.num_hidden), vt(get_batch_size(rbm), rbm.num_visible, 1UL),
            w_grad_b(get_batch_size(rbm), rbm.num_visible, rbm.num_hidden), b_grad_b(get_batch_size(rbm), rbm.num_hidden), c_grad_b(get_batch_size(rbm), rbm.num_visible),
            w_grad(rbm.num_visible, rbm.num_hidden), b_grad(rbm.num_hidden), c_grad(rbm.num_visible),
            w_inc(0,0), b_inc(0), c_inc(0),
            q_global_t(0.0),
            q_local_batch(rbm.num_hidden), q_local_t(rbm.num_hidden, static_cast<weight>(0.0)), q_local_penalty(rbm.num_hidden)
    {
        static_assert(!rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = rbm_traits<rbm_t>::has_momentum(), cpp::enable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t& rbm) :
            v1(get_batch_size(rbm), rbm.num_visible),
            h1_a(get_batch_size(rbm), rbm.num_hidden), h1_s(get_batch_size(rbm), rbm.num_hidden),
            v2_a(get_batch_size(rbm), rbm.num_visible), v2_s(get_batch_size(rbm), rbm.num_visible),
            h2_a(get_batch_size(rbm), rbm.num_hidden), h2_s(get_batch_size(rbm), rbm.num_hidden),
            ht(get_batch_size(rbm), 1UL, rbm.num_hidden), vt(get_batch_size(rbm), rbm.num_visible, 1UL),
            w_grad_b(get_batch_size(rbm), rbm.num_visible, rbm.num_hidden), b_grad_b(get_batch_size(rbm), rbm.num_hidden), c_grad_b(get_batch_size(rbm), rbm.num_visible),
            w_grad(rbm.num_visible, rbm.num_hidden), b_grad(rbm.num_hidden), c_grad(rbm.num_visible),
            w_inc(rbm.num_visible, rbm.num_hidden, static_cast<weight>(0.0)), b_inc(rbm.num_hidden, static_cast<weight>(0.0)), c_inc(rbm.num_visible, static_cast<weight>(0.0)),
            q_global_t(0.0), q_local_batch(rbm.num_hidden), q_local_t(rbm.num_hidden, static_cast<weight>(0.0)), q_local_penalty(rbm.num_hidden)
    {
        static_assert(rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update_weights(RBM& rbm){
        update_weights_normal(rbm, *this);
    }
};

/*!
 * \brief Specialization of base_cd_trainer for Convolutional RBM.
 *
 * This class provides update_weights which applies the gradients to the RBM.
 */
template<typename RBM>
struct base_cd_trainer<RBM, std::enable_if_t<rbm_traits<RBM>::is_convolutional()>> : base_trainer<RBM> {
    typedef RBM rbm_t;

    static constexpr const auto K = rbm_t::K;
    static constexpr const auto NC = rbm_t::NC;
    static constexpr const auto NV = rbm_t::NV;
    static constexpr const auto NH = rbm_t::NH;
    static constexpr const auto NW = rbm_t::NW;

    typedef typename rbm_t::weight weight;

    //Gradients
    etl::fast_matrix<weight, NC, K, NW, NW> w_grad;  //Gradients of shared weights
    etl::fast_vector<weight, K> b_grad;              //Gradients of hidden biases bk
    etl::fast_vector<weight, NC> c_grad;             //Visible gradient

    //{{{ Momentum

    etl::fast_matrix<weight, NC, K, NW, NW> w_inc;
    etl::fast_vector<weight, K> b_inc;
    etl::fast_vector<weight, NC> c_inc;

    //}}} Momentum end

    //{{{ Sparsity

    weight q_global_batch;
    weight q_global_t;

    etl::fast_matrix<weight, K, NH, NH> q_local_batch;
    etl::fast_matrix<weight, K, NH, NH> q_local_t;
    etl::fast_matrix<weight, K, NH, NH> q_local_penalty;

    //}}} Sparsity end

    //{{{ Sparsity biases

    etl::fast_matrix<weight, NC, K, NW, NW> w_bias;
    etl::fast_vector<weight, K> b_bias;
    etl::fast_vector<weight, NC> c_bias;

    //}}} Sparsity biases end

    template<bool M = rbm_traits<rbm_t>::has_momentum(), cpp::disable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t&) :
            q_global_t(0.0), q_local_t(0.0),
            w_bias(0.0), b_bias(0.0), c_bias(0.0) {
        static_assert(!rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = rbm_traits<rbm_t>::has_momentum(), cpp::enable_if_u<M> = cpp::detail::dummy>
    base_cd_trainer(rbm_t&) :
            w_inc(0.0), b_inc(0.0), c_inc(0.0),
            q_global_t(0.0), q_local_t(0.0),
            w_bias(0.0), b_bias(0.0), c_bias(0.0) {
        static_assert(rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update_weights(RBM& rbm){
        //Penalty to be applied to weights and hidden biases
        weight w_penalty = 0.0;
        weight h_penalty = 0.0;
        weight v_penalty = 0.0;

        //Global sparsity method
        if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::GLOBAL_TARGET){
            auto decay_rate = rbm.decay_rate;
            auto p = rbm.sparsity_target;
            auto cost = rbm.sparsity_cost;

            q_global_t = decay_rate * q_global_t + (1.0 - decay_rate) * q_global_batch;

            w_penalty = h_penalty = cost * (q_global_t - p);
        }

        //Apply L1/L2 regularization and penalties to the biases

        base_trainer<RBM>::update_grad(w_grad, rbm.w, rbm, w_decay(rbm_traits<rbm_t>::decay()), w_penalty);
        base_trainer<RBM>::update_grad(b_grad, rbm.b, rbm, b_decay(rbm_traits<rbm_t>::decay()), h_penalty);
        base_trainer<RBM>::update_grad(c_grad, rbm.c, rbm, b_decay(rbm_traits<rbm_t>::decay()), v_penalty);

        //Local sparsity method
        if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
            auto decay_rate = rbm.decay_rate;
            auto p = rbm.sparsity_target;
            auto cost = rbm.sparsity_cost;

            q_local_t = decay_rate * q_local_t + (1.0 - decay_rate) * q_local_batch;
            q_local_penalty = cost * (q_local_t - p);

            for(std::size_t k = 0; k < K; ++k){
                //TODO Ideally, the loop should be removed and the
                //update be done diretly on the base matrix

                //????
                auto h_k_penalty = sum(q_local_penalty(k));
                w_grad(k) -= h_k_penalty;
                b_grad(k) -= h_k_penalty;
            }
        }

        if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::LEE){
            w_grad -= rbm.pbias_lambda * w_bias;
            b_grad -= rbm.pbias_lambda * b_bias;
            c_grad -= rbm.pbias_lambda * c_bias;
        }

        //Apply momentum and learning rate
        if(rbm_traits<rbm_t>::has_momentum()){
            auto momentum = rbm.momentum;
            auto eps = rbm.learning_rate;

            w_inc = momentum * w_inc + eps * w_grad;
            b_inc = momentum * b_inc + eps * b_grad;
            c_inc = momentum * c_inc + eps * c_grad;
        }
        //Apply learning rate
        else {
            auto eps = rbm.learning_rate;

            w_grad *= eps;
            b_grad *= eps;
            c_grad *= eps;
        }

        //Update the weights and biases
        //with the final gradients (if not momentum, these are the real gradients)
        rbm.w += base_trainer<rbm_t>::get_fgrad(w_grad, w_inc);
        rbm.b += base_trainer<rbm_t>::get_fgrad(b_grad, b_inc);
        rbm.c += base_trainer<rbm_t>::get_fgrad(c_grad, c_inc);

        //Check for NaN
        nan_check_deep(rbm.w);
        nan_check_deep(rbm.b);
        nan_check_deep(rbm.c);
    }
};

template<typename RBM, typename C, cpp::enable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
auto reshape_nv1(RBM& rbm, C&& container){
    return etl::reshape(container, rbm.num_visible, 1);
}

template<typename RBM, typename C, cpp::disable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
auto reshape_nv1(RBM&, C&& container){
    return etl::reshape<RBM::num_visible, 1>(container);
}

template<typename RBM, typename C, cpp::enable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
auto reshape_1nh(RBM& rbm, C&& container){
    return etl::reshape(container, 1, rbm.num_hidden);
}

template<typename RBM, typename C, cpp::disable_if_u<rbm_traits<RBM>::is_dynamic()> = cpp::detail::dummy>
auto reshape_1nh(RBM&, C&& container){
    return etl::reshape<1, RBM::num_hidden>(container);
}

template<bool Persistent, std::size_t K, typename T, typename RBM, typename Trainer, typename M>
void train_normal(const dll::batch<T>& batch, rbm_training_context& context, RBM& rbm, Trainer& t, M& t1, M& t2){
    cpp_assert(batch.size() > 0, "Invalid batch size");
    cpp_assert(batch.size() <= get_batch_size(rbm), "Invalid batch size");
    cpp_assert(batch.begin()->size() == input_size(rbm), "The size of the training sample must match visible units");

    using namespace etl;
    using rbm_t = RBM;

    maybe_parallel_foreach_i(t.pool, batch.begin(), batch.end(), [&](const auto& items, std::size_t i){
        //Give input to RBM
        t.v1(i) = items;

        //First step
        rbm.activate_hidden(t.h1_a(i), t.h1_s(i), t.v1(i), t.v1(i), t.ht(i));

        if(Persistent && t.init){
            t.p_h_a(i) = t.h1_a(i);
            t.p_h_s(i) = t.h1_s(i);
        }

        //CD-1
        if(Persistent){
            rbm.activate_visible(t.p_h_a(i), t.p_h_s(i), t.v2_a(i), t.v2_s(i), t.vt(i));
            rbm.activate_hidden(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.ht(i));
        } else {
            rbm.activate_visible(t.h1_a(i), t.h1_s(i), t.v2_a(i), t.v2_s(i), t.vt(i));
            rbm.activate_hidden(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.ht(i));
        }

        //CD-k
        for(std::size_t k = 1; k < K; ++k){
            rbm.activate_visible(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.vt(i));
            rbm.activate_hidden(t.h2_a(i), t.h2_s(i), t.v2_a(i), t.v2_s(i), t.ht(i));
        }

        if(Persistent){
            t.p_h_a(i) = t.h2_a(i);
            t.p_h_s(i) = t.h2_s(i);
        }

        t.w_grad_b(i) = mmul(reshape_nv1(rbm, t.v1(i)), reshape_1nh(rbm, t.h1_a(i)), t1(i))
                      - mmul(reshape_nv1(rbm, t.v2_a(i)), reshape_1nh(rbm, t.h2_a(i)), t2(i));
        t.b_grad_b(i) = t.h1_a(i) - t.h2_a(i);
        t.c_grad_b(i) = t.v1(i) - t.v2_a(i);
    });

    if(Persistent){
        t.init = false;
    }

    context.reconstruction_error += mean((t.v1 - t.v2_a) * (t.v1 - t.v2_a));

    //Keep only the mean of the gradients
    t.w_grad = mean_l(t.w_grad_b);
    t.b_grad = mean_l(t.b_grad_b);
    t.c_grad = mean_l(t.c_grad_b);

    nan_check_deep_3(t.w_grad, t.b_grad, t.c_grad);

    //Compute the mean activation probabilities
    t.q_global_batch = mean(t.h2_a);//n_samples * num_hidden(rbm);

    if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
        t.q_local_batch = mean_l(t.h2_a);
    }

    //Accumulate the sparsity
    context.sparsity += t.q_global_batch;

    //Update the weights and biases based on the gradients
    t.update_weights(rbm);
}

/*!
 * \brief Contrastive divergence trainer for RBM.
 */
template<std::size_t N, typename RBM, typename Enable = void>
struct cd_trainer : base_cd_trainer<RBM> {
    static_assert(N > 0, "CD-0 is not a valid training method");

    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    static constexpr const auto batch_size = rbm_traits<rbm_t>::batch_size();

    rbm_t& rbm;

    //These fields are fake, just to avoid duplicating code and using TMP
    etl::fast_matrix<weight, batch_size, rbm_t::num_hidden> p_h_a;
    etl::fast_matrix<weight, batch_size, rbm_t::num_hidden> p_h_s;

    bool init = true;

    cd_trainer(rbm_t& rbm) : base_cd_trainer<rbm_t>(rbm), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    void train_batch(const dll::batch<T>& batch, rbm_training_context& context){
        static etl::fast_matrix<weight, batch_size, rbm_t::num_visible, rbm_t::num_hidden> t1;
        static etl::fast_matrix<weight, batch_size, rbm_t::num_visible, rbm_t::num_hidden> t2;

        train_normal<false, N>(batch, context, rbm, *this, t1, t2);
    }

    static std::string name(){
        return "Contrastive Divergence";
    }
};

/*!
 * \brief Contrastive divergence trainer for dynamic RBM.
 */
template<std::size_t N, typename RBM>
struct cd_trainer<N, RBM, std::enable_if_t<rbm_traits<RBM>::is_dynamic()>> : base_cd_trainer<RBM> {
    static_assert(N > 0, "CD-0 is not a valid training method");

    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    rbm_t& rbm;

    //These fields are fake, just to avoid duplicating code and using TMP
    etl::dyn_matrix<weight> p_h_a;
    etl::dyn_matrix<weight> p_h_s;

    bool init = true;

    cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(rbm), rbm(rbm), p_h_a(1,1), p_h_s(1,1) {
        //Nothing else to init here
    }

    template<typename T>
    void train_batch(const dll::batch<T>& batch, rbm_training_context& context){
        static etl::dyn_matrix<weight, 3> t1(get_batch_size(rbm), rbm.num_visible, rbm.num_hidden);
        static etl::dyn_matrix<weight, 3> t2(get_batch_size(rbm), rbm.num_visible, rbm.num_hidden);

        train_normal<false, N>(batch, context, rbm, *this, t1, t2);
    }

    static std::string name(){
        return "Contrastive Divergence";
    }
};

/*!
 * \brief Contrastive Divergence trainer for convolutional RBM
 */
template<std::size_t N, typename RBM>
struct cd_trainer<N, RBM, std::enable_if_t<rbm_traits<RBM>::is_convolutional()>> : base_cd_trainer<RBM> {
private:
    static_assert(N > 0, "CD-0 is not a valid training method");

    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    using base_cd_trainer<RBM>::K;
    using base_cd_trainer<RBM>::NW;
    using base_cd_trainer<RBM>::NH;
    using base_cd_trainer<RBM>::NV;
    using base_cd_trainer<RBM>::NC;

    using base_cd_trainer<RBM>::w_grad;
    using base_cd_trainer<RBM>::b_grad;
    using base_cd_trainer<RBM>::c_grad;

    using base_cd_trainer<RBM>::q_global_batch;
    using base_cd_trainer<RBM>::q_local_batch;

    using base_cd_trainer<RBM>::w_bias;
    using base_cd_trainer<RBM>::b_bias;
    using base_cd_trainer<RBM>::c_bias;

    etl::fast_matrix<weight, NC, NV, NV> c_grad_org;

    etl::fast_matrix<weight, NC, K, NW, NW>  w_pos;
    etl::fast_matrix<weight, NC, K, NW, NW>  w_neg;

    rbm_t& rbm;

public:
    cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(rbm), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    void train_batch(const dll::batch<T>& batch, rbm_training_context& context){
        cpp_assert(batch.size() > 0, "Invalid batch size");
        cpp_assert(batch.size() <= get_batch_size(rbm), "Invalid batch size");
        cpp_assert(batch.begin()->size() == input_size(rbm), "The size of the training sample must match visible units");

        //Size of a minibatch
        auto n_samples = static_cast<weight>(batch.size());

        //Clear the gradients
        w_grad = 0.0;
        b_grad = 0.0;
        c_grad_org = 0.0;

        //Reset mean activation probability
        q_global_batch = 0.0;

        if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
            q_local_batch = 0.0;
        }

        if(rbm_traits<rbm_t>::bias_mode() == bias_mode::SIMPLE){
            b_bias = 0.0;
        }

        for(auto& items : batch){
            rbm.v1 = items;

            //First step
            rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);

            //CD-1
            rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);
            rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);

            //CD-k
            for(std::size_t n = 1; n < N; ++n){
                rbm.activate_visible(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
                rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
            }

            //Compute gradients

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(std::size_t k = 0; k < K; ++k){
                    etl::convolve_2d_valid(rbm.v1(channel), fflip(rbm.h1_a(k)), w_pos(channel)(k));
                    etl::convolve_2d_valid(rbm.v2_a(channel), fflip(rbm.h2_a(k)), w_neg(channel)(k));
                }
            }

            w_grad += w_pos - w_neg;

            for(std::size_t k = 0; k < K; ++k){
                //TODO Find if mean or sum
                b_grad(k) += sum(rbm.h1_a(k) - rbm.h2_a(k));
            }

            c_grad_org += rbm.v1 - rbm.v2_a;

            context.reconstruction_error += mean((rbm.v1 - rbm.v2_a) * (rbm.v1 - rbm.v2_a));

            q_global_batch += sum(rbm.h2_a);

            if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
                q_local_batch += rbm.h2_a;
            }

            //Compute the biases for sparsity
            if(rbm_traits<rbm_t>::bias_mode() == bias_mode::SIMPLE){
                for(std::size_t k = 0; k < K; ++k){
                    b_bias(k) += mean(rbm.h2_a(k));
                }
            }
        }

        //Keep only the mean of the gradients
        w_grad /= n_samples;
        b_grad /= n_samples;
        c_grad_org /= n_samples;

        nan_check_deep(w_grad);
        nan_check_deep(b_grad);
        nan_check_deep(c_grad_org);

        c_grad = mean(c_grad_org);

        //Compute the mean activation probabilities
        q_global_batch /= n_samples * K * NH * NH;

        if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
            q_local_batch /= n_samples;
        }

        if(rbm_traits<rbm_t>::bias_mode() == bias_mode::SIMPLE){
            b_bias = b_bias / n_samples - rbm.pbias;
        }

        //Accumulate the sparsity
        context.sparsity += q_global_batch;

        //Update the weights and biases based on the gradients
        this->update_weights(rbm);
    }

    static std::string name(){
        return "Contrastive Divergence (convolutional)";
    }
};

/*!
 * \brief Persistent Contrastive Divergence Trainer for RBM.
 */
template<std::size_t K, typename RBM, typename Enable = void>
struct persistent_cd_trainer : base_cd_trainer<RBM> {
    static_assert(K > 0, "PCD-0 is not a valid training method");

    typedef RBM rbm_t;
    typedef typename rbm_t::weight weight;

    static constexpr const auto batch_size = rbm_traits<rbm_t>::batch_size();

    etl::fast_matrix<weight, batch_size, rbm_t::num_hidden> p_h_a;
    etl::fast_matrix<weight, batch_size, rbm_t::num_hidden> p_h_s;

    bool init = true;

    rbm_t& rbm;

    persistent_cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(rbm), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    void train_batch(const dll::batch<T>& batch, rbm_training_context& context){
        static etl::fast_matrix<weight, batch_size, rbm_t::num_visible, rbm_t::num_hidden> t1;
        static etl::fast_matrix<weight, batch_size, rbm_t::num_visible, rbm_t::num_hidden> t2;

        train_normal<true, K>(batch, context, rbm, *this, t1, t2);
    }

    static std::string name(){
        return "Persistent Contrastive Divergence";
    }
};

/*!
 * \brief Persistent Contrastive Divergence Trainer for RBM.
 */
template<std::size_t K, typename RBM>
struct persistent_cd_trainer<K, RBM, std::enable_if_t<rbm_traits<RBM>::is_dynamic()>> : base_cd_trainer<RBM> {
    static_assert(K > 0, "PCD-0 is not a valid training method");

    typedef RBM rbm_t;
    typedef typename rbm_t::weight weight;

    rbm_t& rbm;

    etl::dyn_matrix<weight> p_h_a;
    etl::dyn_matrix<weight> p_h_s;

    bool init = true;

    persistent_cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(rbm), rbm(rbm),
            p_h_a(get_batch_size(rbm), rbm.num_hidden), p_h_s(get_batch_size(rbm), rbm.num_hidden)  {
        //Nothing else to init
    }

    template<typename T>
    void train_batch(const dll::batch<T>& batch, rbm_training_context& context){
        static etl::dyn_matrix<weight, 3> t1(get_batch_size(rbm), rbm.num_visible, rbm.num_hidden);
        static etl::dyn_matrix<weight, 3> t2(get_batch_size(rbm), rbm.num_visible, rbm.num_hidden);

        train_normal<true, K>(batch, context, rbm, *this, t1, t2);
    }

    static std::string name(){
        return "Persistent Contrastive Divergence";
    }
};


/*!
 * \brief Specialization of persistent_cd_trainer for Convolutional RBM.
 */
template<std::size_t N, typename RBM>
struct persistent_cd_trainer<N, RBM, std::enable_if_t<rbm_traits<RBM>::is_convolutional()>> : base_cd_trainer<RBM> {
private:
    static_assert(N > 0, "PCD-0 is not a valid training method");

    typedef RBM rbm_t;
    typedef typename rbm_t::weight weight;

    using base_cd_trainer<RBM>::NC;
    using base_cd_trainer<RBM>::NW;
    using base_cd_trainer<RBM>::NH;
    using base_cd_trainer<RBM>::NV;
    using base_cd_trainer<RBM>::K;

    using base_cd_trainer<RBM>::w_grad;
    using base_cd_trainer<RBM>::b_grad;
    using base_cd_trainer<RBM>::c_grad;

    etl::fast_matrix<weight, NC, NV, NV> c_grad_org;

    using base_cd_trainer<RBM>::q_global_batch;
    using base_cd_trainer<RBM>::q_local_batch;

    etl::fast_matrix<weight, NC, K, NW, NW> w_pos;
    etl::fast_matrix<weight, NC, K, NW, NW> w_neg;

    std::vector<etl::fast_matrix<weight, K, NH, NH>> p_h_a;
    std::vector<etl::fast_matrix<weight, K, NH, NH>> p_h_s;

    bool init = true;

    rbm_t& rbm;

public:
    persistent_cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(rbm),
            p_h_a(get_batch_size(rbm)),
            p_h_s(get_batch_size(rbm)), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    void train_batch(const dll::batch<T>& batch, rbm_training_context& context){
        cpp_assert(batch.size() > 0, "Invalid batch size");
        cpp_assert(batch.size() <= get_batch_size(rbm), "Invalid batch size");
        cpp_assert(batch.begin()->size() == input_size(rbm), "The size of the training sample must match visible units");

        //Size of a minibatch
        auto n_samples = static_cast<weight>(batch.size());

        //Clear the gradients
        w_grad = 0.0;
        b_grad = 0.0;
        c_grad_org = 0.0;

        //Reset mean activation probability if necessary
        q_global_batch = 0.0;

        if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
            q_local_batch = 0.0;
        }

        auto it = batch.begin();
        auto end = batch.end();

        std::size_t i = 0;
        while(it != end){
            auto& items = *it;

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
            for(std::size_t k = 1; k < N; ++k){
                rbm.activate_visible(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
                rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
            }

            p_h_a[i] = rbm.h2_a;
            p_h_s[i] = rbm.h2_s;

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(std::size_t k = 0; k < K; ++k){
                    etl::convolve_2d_valid(rbm.v1(channel), fflip(rbm.h1_a(k)), w_pos(channel)(k));
                    etl::convolve_2d_valid(rbm.v2_a(channel), fflip(rbm.h2_a(k)), w_neg(channel)(k));
                }
            }

            w_grad += w_pos - w_neg;

            for(std::size_t k = 0; k < K; ++k){
                b_grad(k) += mean(rbm.h1_a(k) - rbm.h2_a(k));
            }

            c_grad_org += rbm.v1 - rbm.v2_a;

            context.reconstruction_error += mean((rbm.v1 - rbm.v2_a) * (rbm.v1 - rbm.v2_a));

            q_global_batch += sum(rbm.h2_a);

            if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
                q_local_batch += rbm.h2_a;
            }

            ++it;
            ++i;
        }

        init = false;

        //Keep only the mean of the gradients
        w_grad /= n_samples;
        b_grad /= n_samples;
        c_grad_org /= n_samples;

        nan_check_deep(w_grad);
        nan_check_deep(b_grad);
        nan_check_deep(c_grad_org);

        c_grad = mean(c_grad_org);

        //Compute the mean activation probabilities
        q_global_batch /= n_samples * K * NH * NH;

        if(rbm_traits<rbm_t>::sparsity_method() == sparsity_method::LOCAL_TARGET){
            q_local_batch /= n_samples;
        }

        //Accumulate the sparsity
        context.sparsity += q_global_batch;

        //Update the weights and biases based on the gradients
        this->update_weights(rbm);
    }

    static std::string name(){
        return "Persistent Contrastive Divergence (convolutional)";
    }
};

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

} //end of dbn namespace

#endif
