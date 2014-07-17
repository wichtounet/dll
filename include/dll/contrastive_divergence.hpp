//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*! \file Contrastive Divergence Implementations */

#ifndef DBN_CONTRASTIVE_DIVERGENCE_HPP
#define DBN_CONTRASTIVE_DIVERGENCE_HPP

#include "etl/fast_matrix.hpp"
#include "etl/fast_vector.hpp"
#include "etl/convolution.hpp"

#include "assert.hpp"
#include "batch.hpp"
#include "decay_type.hpp"
#include "rbm_traits.hpp"

namespace dll {

//Sign for scalars
inline double sign(double v){
    return v == 0.0 ? 0.0 : (v > 0.0 ? 1.0 : -1.0);
}

/*!
 * \brief Indicates the type of decay that is to be applied to weights
 * \param t The RBM weight decay type.
 * \return one of L1,L2,NONE
 */
constexpr decay_type w_decay(decay_type t){
    if(t == decay_type::L1 || t == decay_type::L1_FULL){
        return decay_type::L1;
    } else if(t == decay_type::L2 || t == decay_type::L2_FULL){
        return decay_type::L2;
    } else {
        return decay_type::NONE;
    }
}

/*!
 * \brief Indicates the type of decay that is to be applied to biases
 * \param t The RBM weight decay type.
 * \return one of L1,L2,NONE
 */
constexpr decay_type b_decay(decay_type t){
    if(t == decay_type::L1_FULL){
        return decay_type::L1;
    } else if(t == decay_type::L2_FULL){
        return decay_type::L2;
    } else {
        return decay_type::NONE;
    }
}

/*!
 * \brief Base class for all standard trainer
 */
template<typename RBM>
struct base_trainer {
    typedef RBM rbm_t;

    template<typename T1, typename T2, bool M = rbm_traits<rbm_t>::has_momentum(), enable_if_u<M> = ::detail::dummy>
    T2& get_fgrad(T1& , T2& inc){
        return inc;
    }

    template<typename T1, typename T2, bool M = rbm_traits<rbm_t>::has_momentum(), disable_if_u<M> = ::detail::dummy>
    T1& get_fgrad(T1& grad, T2& ){
        return grad;
    }

    template<typename V, typename G>
    void update(V& value, const G& grad, const RBM& rbm, decay_type decay, double penalty){
        if(decay == decay_type::L1){
            value += rbm.learning_rate * grad - rbm.weight_cost * abs(value) - penalty;
        } else if(decay == decay_type::L2){
            value += rbm.learning_rate * grad - rbm.weight_cost * value - penalty;
        } else {
            value += rbm.learning_rate * grad - penalty;
        }
    }
};

/*!
 * \brief Base class for all Contrastive Divergence Trainer.
 *
 * This class provides update_weights which applies the gradients to the RBM.
 */
template<typename RBM, typename Enable = void>
struct base_cd_trainer : base_trainer<RBM> {
    typedef RBM rbm_t;

    static constexpr const auto num_hidden = rbm_t::num_hidden;
    static constexpr const auto num_visible = rbm_t::num_visible;

    typedef typename rbm_t::weight weight;

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

    weight q_batch;
    weight q_t;

    //}}} Sparsity end

    template<bool M = rbm_traits<rbm_t>::has_momentum(), disable_if_u<M> = ::detail::dummy>
    base_cd_trainer() : q_t(0.0) {
        static_assert(!rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = rbm_traits<rbm_t>::has_momentum(), enable_if_u<M> = ::detail::dummy>
    base_cd_trainer() : w_inc(0.0), b_inc(0.0), c_inc(0.0), q_t(0.0) {
        static_assert(rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update_weights(RBM& rbm){
        //Update momentum gradients
        if(rbm_traits<rbm_t>::has_momentum()){
            auto momentum = rbm.momentum;

            w_inc = momentum * w_inc + (1 - momentum) * w_grad;
            b_inc = momentum * b_inc + (1 - momentum) * b_grad;
            c_inc = momentum * c_inc + (1 - momentum) * c_grad;
        }

        //Penalty to be applied to weights and hidden biases
        weight h_penalty = 0.0;

        //Update sparsity
        if(rbm_traits<rbm_t>::has_sparsity()){
            auto decay_rate = rbm.decay_rate;
            auto p = rbm.sparsity_target;
            auto cost = rbm.sparsity_cost;

            q_t = decay_rate * q_t + (1.0 - decay_rate) * q_batch;

            h_penalty = cost * (q_t - p);
        }

        //The final gradients;
        const auto& w_fgrad = base_trainer<RBM>::get_fgrad(w_grad, w_inc);
        const auto& b_fgrad = base_trainer<RBM>::get_fgrad(b_grad, b_inc);
        const auto& c_fgrad = base_trainer<RBM>::get_fgrad(c_grad, c_inc);

        //Weight decay is applied on biases only on demand
        //Note: According to G. Hinton, Weight Decay should not be applied to
        //biases by default due to their limited number and therefore their weak
        //contribution to overfitting

        //Update weights and biases

        base_trainer<RBM>::update(rbm.w, w_fgrad, rbm, w_decay(rbm_traits<rbm_t>::decay()), h_penalty);
        base_trainer<RBM>::update(rbm.b, b_fgrad, rbm, b_decay(rbm_traits<rbm_t>::decay()), h_penalty);
        base_trainer<RBM>::update(rbm.c, c_fgrad, rbm, b_decay(rbm_traits<rbm_t>::decay()), 0.0);

        //Check for NaN
        nan_check_deep_3(rbm.w, rbm.b, rbm.c);
    }
};

/*!
 * \brief Specialization of base_cd_trainer for Convolutional RBM.
 *
 * This class provides update_weights which applies the gradients to the RBM.
 */
template<typename RBM>
struct base_cd_trainer<RBM, enable_if_t<rbm_traits<RBM>::is_convolutional()>> : base_trainer<RBM> {
    typedef RBM rbm_t;

    static constexpr const auto K = rbm_t::K;
    static constexpr const auto NV = rbm_t::NV;
    static constexpr const auto NH = rbm_t::NH;
    static constexpr const auto NW = rbm_t::NW;

    typedef typename rbm_t::weight weight;

    //Gradients
    etl::fast_vector<etl::fast_matrix<weight, NW, NW>, K>  w_grad;      //Gradients of shared weights
    etl::fast_vector<weight, K> b_grad;                                 //Gradients of hidden biases bk

    weight c_grad;                                      //Visible gradient

    //{{{ Momentum

    etl::fast_vector<etl::fast_matrix<weight, NW, NW>, K>  w_inc;
    etl::fast_vector<weight, K> b_inc;
    weight c_inc;

    //}}} Momentum end

    //{{{ Sparsity

    weight q_batch;
    weight q_t;

    //}}} Sparsity end

    template<bool M = rbm_traits<rbm_t>::has_momentum(), disable_if_u<M> = ::detail::dummy>
    base_cd_trainer() : q_t(0.0) {
        static_assert(!rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = rbm_traits<rbm_t>::has_momentum(), enable_if_u<M> = ::detail::dummy>
    base_cd_trainer() : w_inc(0.0), b_inc(0.0), c_inc(0.0), q_t(0.0) {
        static_assert(rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used with momentum support");
    }

    void update_weights(RBM& rbm){
        //Update momentum gradients
        if(rbm_traits<rbm_t>::has_momentum()){
            auto momentum = rbm.momentum;

            for(std::size_t k = 0; k < K; ++k){
                w_inc(k) = momentum * w_inc(k) + (1 - momentum) * w_grad(k);
            }

            b_inc = momentum * b_inc + (1 - momentum) * b_grad;
            c_inc = momentum * c_inc + (1 - momentum) * c_grad;
        }

        //Penalty to be applied to weights and hidden biases
        weight h_penalty = 0.0;

        //Update sparsity
        if(rbm_traits<rbm_t>::has_sparsity()){
            auto decay_rate = rbm.decay_rate;
            auto p = rbm.sparsity_target;
            auto cost = rbm.sparsity_cost;

            q_t = decay_rate * q_t + (1.0 - decay_rate) * q_batch;

            h_penalty = cost * (q_t - p);
        }

        //The final gradients;
        const auto& w_fgrad = base_trainer<RBM>::get_fgrad(w_grad, w_inc);
        const auto& b_fgrad = base_trainer<RBM>::get_fgrad(b_grad, b_inc);
        const auto& c_fgrad = base_trainer<RBM>::get_fgrad(c_grad, c_inc);

        //Weight decay is applied on biases only on demand
        //Note: According to G. Hinton, Weight Decay should not be applied to
        //biases by default due to their limited number and therefore their weak
        //contribution to overfitting

        //Update weights and biases

        for(std::size_t k = 0; k < K; ++k){
            //TODO Ideally, the loop should be removed and the
            //update be done diretly on rbm.w
            base_trainer<RBM>::update(rbm.w(k), w_fgrad(k), rbm, w_decay(rbm_traits<rbm_t>::decay()), h_penalty);
        }

        base_trainer<RBM>::update(rbm.b, b_fgrad, rbm, b_decay(rbm_traits<rbm_t>::decay()), h_penalty);
        base_trainer<RBM>::update(rbm.c, c_fgrad, rbm, b_decay(rbm_traits<rbm_t>::decay()), 0.0);

        //Check for NaN
        nan_check_deep_deep(rbm.w);
        nan_check_deep(rbm.b);
        nan_check(rbm.c);
    }
};

/*!
 * \brief Contrastive divergence trainer for RBM.
 */
template<std::size_t N, typename RBM, typename Enable = void>
struct cd_trainer : base_cd_trainer<RBM> {
private:
    static_assert(N > 0, "CD-0 is not a valid training method");

    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    using base_cd_trainer<RBM>::num_visible;
    using base_cd_trainer<RBM>::num_hidden;

    using base_cd_trainer<RBM>::w_grad;
    using base_cd_trainer<RBM>::b_grad;
    using base_cd_trainer<RBM>::c_grad;

    using base_cd_trainer<RBM>::q_batch;

    rbm_t& rbm;

public:
    cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    weight train_batch(const dll::batch<T>& batch){
        dll_assert(batch.size() <= static_cast<typename dll::batch<T>::size_type>(rbm_traits<rbm_t>::batch_size()), "Invalid size");
        dll_assert(batch[0].size() == num_visible, "The size of the training sample must match visible units");

        //Size of a minibatch
        auto n_samples = static_cast<weight>(batch.size());

        //Clear the gradients
        w_grad = 0.0;
        b_grad = 0.0;
        c_grad = 0.0;

        //Reset mean activation probability if necessary
        if(rbm_traits<rbm_t>::has_sparsity()){
            q_batch = 0.0;
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

            for(size_t i = 0; i < num_visible; ++i){
                for(size_t j = 0; j < num_hidden; ++j){
                    w_grad(i, j) += rbm.h1_a(j) * rbm.v1(i) - rbm.h2_a(j) * rbm.v2_a(i);
                }
            }

            b_grad += rbm.h1_a - rbm.h2_a;
            c_grad += rbm.v1 - rbm.v2_a;

            if(rbm_traits<rbm_t>::has_sparsity()){
                q_batch += sum(rbm.h2_a);
            }
        }

        //Keep only the mean of the gradients
        w_grad /= n_samples;
        b_grad /= n_samples;
        c_grad /= n_samples;

        //Compute the mean activation probabilities
        if(rbm_traits<rbm_t>::has_sparsity()){
            q_batch /= n_samples * num_hidden;
        }

        nan_check_deep_3(w_grad, b_grad, c_grad);

        //Update the weights and biases based on the gradients
        this->update_weights(rbm);

        //Return the reconstruction error
        return mean(c_grad * c_grad);
    }
};

/*!
 * \brief Specialization of cd_trainer for Convolutional RBM.
 */
template<std::size_t N, typename RBM>
struct cd_trainer<N, RBM, enable_if_t<rbm_traits<RBM>::is_convolutional()>> : base_cd_trainer<RBM> {
private:
    static_assert(N > 0, "CD-0 is not a valid training method");

    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    using base_cd_trainer<RBM>::K;
    using base_cd_trainer<RBM>::NW;
    using base_cd_trainer<RBM>::NH;
    using base_cd_trainer<RBM>::NV;

    using base_cd_trainer<RBM>::w_grad;
    using base_cd_trainer<RBM>::b_grad;
    using base_cd_trainer<RBM>::c_grad;

    etl::fast_matrix<weight, NV, NV> c_grad_org;

    using base_cd_trainer<RBM>::q_batch;

    etl::fast_vector<etl::fast_matrix<weight, NW, NW>, K>  w_pos;
    etl::fast_vector<etl::fast_matrix<weight, NW, NW>, K>  w_neg;

    rbm_t& rbm;

public:
    cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    weight train_batch(const dll::batch<T>& batch){
        dll_assert(batch.size() <= static_cast<typename dll::batch<T>::size_type>(rbm_traits<rbm_t>::batch_size()), "Invalid size");
        dll_assert(batch[0].size() == rbm_t::NV * rbm_t::NV, "The size of the training sample must match visible units");

        //Size of a minibatch
        auto n_samples = static_cast<weight>(batch.size());

        //Clear the gradients
        w_grad = 0.0;
        b_grad = 0.0;
        c_grad_org = 0.0;

        //Reset mean activation probability if necessary
        if(rbm_traits<rbm_t>::has_sparsity()){
            q_batch = 0.0;
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

            for(std::size_t k = 0; k < K; ++k){
                etl::convolve_2d_valid(rbm.v1, fflip(rbm.h1_a(k)), w_pos(k));
                etl::convolve_2d_valid(rbm.v2_a, fflip(rbm.h2_a(k)), w_neg(k));

                w_grad(k) += w_pos(k) - w_neg(k);
            }

            for(std::size_t k = 0; k < K; ++k){
                b_grad(k) += mean(rbm.h1_a(k) - rbm.h2_a(k));
            }

            c_grad_org += rbm.v1 - rbm.v2_a;

            if(rbm_traits<rbm_t>::has_sparsity()){
                q_batch += sum(sum(rbm.h2_a));
            }
        }

        //Keep only the mean of the gradients
        w_grad /= n_samples;
        b_grad /= n_samples;
        c_grad_org /= n_samples;

        nan_check_deep_deep(w_grad);
        nan_check_deep(b_grad);
        nan_check_deep(c_grad_org);

        c_grad = mean(c_grad_org);

        //Compute the mean activation probabilities
        if(rbm_traits<rbm_t>::has_sparsity()){
            q_batch /= n_samples * K * NH * NH;
        }

        //Update the weights and biases based on the gradients
        this->update_weights(rbm);

        //Return the reconstruction error
        return mean(c_grad_org * c_grad_org);
    }
};

/*!
 * \brief Persistent Contrastive Divergence Trainer for RBM.
 */
template<std::size_t K, typename RBM, typename Enable = void>
struct persistent_cd_trainer : base_cd_trainer<RBM> {
private:
    static_assert(K > 0, "PCD-0 is not a valid training method");

    typedef RBM rbm_t;
    typedef typename rbm_t::weight weight;

    using base_cd_trainer<RBM>::num_visible;
    using base_cd_trainer<RBM>::num_hidden;

    using base_cd_trainer<RBM>::w_grad;
    using base_cd_trainer<RBM>::b_grad;
    using base_cd_trainer<RBM>::c_grad;

    using base_cd_trainer<RBM>::q_batch;

    std::vector<etl::fast_vector<weight, num_hidden>> p_h_a;
    std::vector<etl::fast_vector<weight, num_hidden>> p_h_s;

    bool init = true;

    rbm_t& rbm;

public:
    persistent_cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(),
            p_h_a(rbm_traits<rbm_t>::batch_size()),
            p_h_s(rbm_traits<rbm_t>::batch_size()), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    weight train_batch(const dll::batch<T>& batch){
        dll_assert(batch.size() <= static_cast<typename dll::batch<T>::size_type>(rbm_traits<rbm_t>::batch_size()), "Invalid size");
        dll_assert(batch[0].size() == num_visible, "The size of the training sample must match visible units");

        //Size of a minibatch
        auto n_samples = static_cast<weight>(batch.size());

        //Clear the gradients
        w_grad = 0.0;
        b_grad = 0.0;
        c_grad = 0.0;

        //Reset mean activation probability if necessary
        if(rbm_traits<rbm_t>::has_sparsity()){
            q_batch = 0.0;
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

            b_grad += rbm.h1_a - rbm.h2_a;
            c_grad += rbm.v1 - rbm.v2_a;

            if(rbm_traits<rbm_t>::has_sparsity()){
                q_batch += sum(rbm.h2_a);
            }
        }

        init = false;

        //Keep only the mean of the gradients
        w_grad /= n_samples;
        b_grad /= n_samples;
        c_grad /= n_samples;

        nan_check_deep_3(w_grad, b_grad, c_grad);

        //Compute the mean activation probabilities
        if(rbm_traits<rbm_t>::has_sparsity()){
            q_batch /= n_samples * num_hidden;
        }

        //Update the weights and biases based on the gradients
        this->update_weights(rbm);

        //Return the reconstruction error
        return mean(c_grad * c_grad);
    }
};

/*!
 * \brief Specialization of persistent_cd_trainer for Convolutional RBM.
 */
template<std::size_t N, typename RBM>
struct persistent_cd_trainer<N, RBM, enable_if_t<rbm_traits<RBM>::is_convolutional()>> : base_cd_trainer<RBM> {
private:
    static_assert(N > 0, "PCD-0 is not a valid training method");

    typedef RBM rbm_t;
    typedef typename rbm_t::weight weight;

    using base_cd_trainer<RBM>::NW;
    using base_cd_trainer<RBM>::NH;
    using base_cd_trainer<RBM>::NV;
    using base_cd_trainer<RBM>::K;

    using base_cd_trainer<RBM>::w_grad;
    using base_cd_trainer<RBM>::b_grad;
    using base_cd_trainer<RBM>::c_grad;

    etl::fast_matrix<weight, NV, NV> c_grad_org;

    using base_cd_trainer<RBM>::q_batch;

    etl::fast_vector<etl::fast_matrix<weight, NW, NW>, K>  w_pos;
    etl::fast_vector<etl::fast_matrix<weight, NW, NW>, K>  w_neg;

    std::vector<etl::fast_vector<etl::fast_matrix<weight, NH, NH>, K>> p_h_a;
    std::vector<etl::fast_vector<etl::fast_matrix<weight, NH, NH>, K>> p_h_s;

    bool init = true;

    rbm_t& rbm;

public:
    persistent_cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(),
            p_h_a(rbm_traits<rbm_t>::batch_size()),
            p_h_s(rbm_traits<rbm_t>::batch_size()), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    weight train_batch(const dll::batch<T>& batch){
        dll_assert(batch.size() <= static_cast<typename dll::batch<T>::size_type>(rbm_traits<rbm_t>::batch_size()), "Invalid size");
        dll_assert(batch[0].size() == NV * NV, "The size of the training sample must match visible units");

        //Size of a minibatch
        auto n_samples = static_cast<weight>(batch.size());

        //Clear the gradients
        w_grad = 0.0;
        b_grad = 0.0;
        c_grad_org = 0.0;

        //Reset mean activation probability if necessary
        if(rbm_traits<rbm_t>::has_sparsity()){
            q_batch = 0.0;
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
            for(std::size_t k = 1; k < N; ++k){
                rbm.activate_visible(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
                rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
            }

            p_h_a[i] = rbm.h2_a;
            p_h_s[i] = rbm.h2_s;

            for(std::size_t k = 0; k < K; ++k){
                etl::convolve_2d_valid(rbm.v1, fflip(rbm.h1_a(k)), w_pos(k));
                etl::convolve_2d_valid(rbm.v2_a, fflip(rbm.h2_a(k)), w_neg(k));

                w_grad(k) += w_pos(k) - w_neg(k);
            }

            for(std::size_t k = 0; k < K; ++k){
                b_grad(k) += mean(rbm.h1_a(k) - rbm.h2_a(k));
            }

            c_grad_org += rbm.v1 - rbm.v2_a;

            if(rbm_traits<rbm_t>::has_sparsity()){
                q_batch += sum(sum(rbm.h2_a));
            }
        }

        init = false;

        //Keep only the mean of the gradients
        w_grad /= n_samples;
        b_grad /= n_samples;
        c_grad_org /= n_samples;

        nan_check_deep_deep(w_grad);
        nan_check_deep(b_grad);
        nan_check_deep(c_grad_org);

        c_grad = mean(c_grad_org);

        //Compute the mean activation probabilities
        if(rbm_traits<rbm_t>::has_sparsity()){
            q_batch /= n_samples * K * NH * NH;
        }

        //Update the weights and biases based on the gradients
        this->update_weights(rbm);

        //Return the reconstruction error
        return mean(c_grad_org * c_grad_org);
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