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

template<typename RBM, typename Trainer>
void update_weights_normal(RBM& rbm, Trainer& t){
    using rbm_t = RBM;

    //Update momentum gradients
    if(rbm_traits<rbm_t>::has_momentum()){
        auto momentum = rbm.momentum;

        t.w_inc = momentum * t.w_inc + (1 - momentum) * t.w_grad;
        t.b_inc = momentum * t.b_inc + (1 - momentum) * t.b_grad;
        t.c_inc = momentum * t.c_inc + (1 - momentum) * t.c_grad;
    }

    //Penalty to be applied to weights and hidden biases
    typename rbm_t::weight h_penalty = 0.0;

    //Update sparsity
    if(rbm_traits<rbm_t>::has_sparsity()){
        auto decay_rate = rbm.decay_rate;
        auto p = rbm.sparsity_target;
        auto cost = rbm.sparsity_cost;

        t.q_t = decay_rate * t.q_t + (1.0 - decay_rate) * t.q_batch;

        h_penalty = cost * (t.q_t - p);
    }

    //The final gradients;
    const auto& w_fgrad = t.get_fgrad(t.w_grad, t.w_inc);
    const auto& b_fgrad = t.get_fgrad(t.b_grad, t.b_inc);
    const auto& c_fgrad = t.get_fgrad(t.c_grad, t.c_inc);

    //Update weights and biases

    t.update(rbm.w, w_fgrad, rbm, w_decay(rbm_traits<rbm_t>::decay()), h_penalty);
    t.update(rbm.b, b_fgrad, rbm, b_decay(rbm_traits<rbm_t>::decay()), h_penalty);
    t.update(rbm.c, c_fgrad, rbm, b_decay(rbm_traits<rbm_t>::decay()), 0.0);

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
    base_cd_trainer(rbm_t&) : q_t(0.0) {
        static_assert(!rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = rbm_traits<rbm_t>::has_momentum(), enable_if_u<M> = ::detail::dummy>
    base_cd_trainer(rbm_t&) : w_inc(0.0), b_inc(0.0), c_inc(0.0), q_t(0.0) {
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
struct base_cd_trainer<RBM, enable_if_t<rbm_traits<RBM>::is_dynamic()>> : base_trainer<RBM> {
    typedef RBM rbm_t;

    typedef typename rbm_t::weight weight;

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

    weight q_batch;
    weight q_t;

    //}}} Sparsity end

    template<bool M = rbm_traits<rbm_t>::has_momentum(), disable_if_u<M> = ::detail::dummy>
    base_cd_trainer(rbm_t& rbm) :
            w_grad(rbm.num_visible, rbm.num_hidden), b_grad(rbm.num_hidden), c_grad(rbm.num_visible),
            w_inc(0,0), b_inc(0), c_inc(0),
            q_t(0.0) {
        static_assert(!rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = rbm_traits<rbm_t>::has_momentum(), enable_if_u<M> = ::detail::dummy>
    base_cd_trainer(rbm_t& rbm) :
            w_grad(rbm.num_visible, rbm.num_hidden), b_grad(rbm.num_hidden), c_grad(rbm.num_visible),
            w_inc(rbm.num_visible, rbm.num_hidden, 0.0), b_inc(rbm.num_hidden, 0.0), c_inc(rbm.num_visible, 0.0),
            q_t(0.0) {
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
    base_cd_trainer(rbm_t&) : q_t(0.0) {
        static_assert(!rbm_traits<rbm_t>::has_momentum(), "This constructor should only be used without momentum support");
    }

    template<bool M = rbm_traits<rbm_t>::has_momentum(), enable_if_u<M> = ::detail::dummy>
    base_cd_trainer(rbm_t&) : w_inc(0.0), b_inc(0.0), c_inc(0.0), q_t(0.0) {
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

template<typename RBM, typename C, enable_if_u<rbm_traits<RBM>::is_dynamic()> = ::detail::dummy>
auto reshape_nv1(RBM& rbm, C& container){
    return etl::reshape(container, rbm.num_visible, 1);
}

template<typename RBM, typename C, disable_if_u<rbm_traits<RBM>::is_dynamic()> = ::detail::dummy>
auto reshape_nv1(RBM&, C& container){
    return etl::reshape<RBM::num_visible, 1>(container);
}

template<typename RBM, typename C, enable_if_u<rbm_traits<RBM>::is_dynamic()> = ::detail::dummy>
auto reshape_1nh(RBM& rbm, C& container){
    return etl::reshape(container, 1, rbm.num_hidden);
}

template<typename RBM, typename C, disable_if_u<rbm_traits<RBM>::is_dynamic()> = ::detail::dummy>
auto reshape_1nh(RBM&, C& container){
    return etl::reshape<1, RBM::num_hidden>(container);
}

template<bool Persistent, std::size_t K, typename T, typename RBM, typename Trainer, typename M>
typename RBM::weight train_normal(const dll::batch<T>& batch, RBM& rbm, Trainer& t, M& t1, M& t2){
    dll_assert(batch.size() <= static_cast<typename dll::batch<T>::size_type>(rbm_traits<RBM>::batch_size()), "Invalid size");
    dll_assert(batch[0].size() == num_visible(rbm), "The size of the training sample must match visible units");

    using namespace etl;
    using rbm_t = RBM;

    //Size of a minibatch
    auto n_samples = static_cast<typename rbm_t::weight>(batch.size());

    //Clear the gradients
    t.w_grad = 0.0;
    t.b_grad = 0.0;
    t.c_grad = 0.0;

    //Reset mean activation probability if necessary
    if(rbm_traits<rbm_t>::has_sparsity()){
        t.q_batch = 0.0;
    }

    for(std::size_t i = 0; i < batch.size(); ++i){
        auto& items = batch[i];

        rbm.v1 = items;

        //First step
        rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);

        if(Persistent && t.init){
            t.p_h_a[i] = rbm.h1_a;
            t.p_h_s[i] = rbm.h1_s;
        }

        //CD-1
        rbm.activate_visible(Persistent ? t.p_h_a[i] : rbm.h1_a, Persistent ? t.p_h_s[i] : rbm.h1_s, rbm.v2_a, rbm.v2_s);
        rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);

        //CD-k
        for(std::size_t k = 1; k < K; ++k){
            rbm.activate_visible(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
            rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);
        }

        if(Persistent){
            t.p_h_a[i] = rbm.h2_a;
            t.p_h_s[i] = rbm.h2_s;
        }

        t.w_grad += mmul(reshape_nv1(rbm, rbm.v1), reshape_1nh(rbm, rbm.h1_a), t1) - mmul(reshape_nv1(rbm, rbm.v2_a), reshape_1nh(rbm, rbm.h2_a), t2);
        t.b_grad += rbm.h1_a - rbm.h2_a;
        t.c_grad += rbm.v1 - rbm.v2_a;

        if(rbm_traits<rbm_t>::has_sparsity()){
            t.q_batch += sum(rbm.h2_a);
        }
    }

    if(Persistent){
        t.init = false;
    }

    //Keep only the mean of the gradients
    t.w_grad /= n_samples;
    t.b_grad /= n_samples;
    t.c_grad /= n_samples;

    nan_check_deep_3(t.w_grad, t.b_grad, t.c_grad);

    //Compute the mean activation probabilities
    if(rbm_traits<rbm_t>::has_sparsity()){
        t.q_batch /= n_samples * num_hidden(rbm);
    }

    //Update the weights and biases based on the gradients
    t.update_weights(rbm);

    //Return the reconstruction error
    return mean(t.c_grad * t.c_grad);
}

/*!
 * \brief Contrastive divergence trainer for RBM.
 */
template<std::size_t N, typename RBM, typename Enable = void>
struct cd_trainer : base_cd_trainer<RBM> {
    static_assert(N > 0, "CD-0 is not a valid training method");

    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    rbm_t& rbm;

    std::vector<etl::fast_vector<weight, rbm_t::num_hidden>> p_h_a;
    std::vector<etl::fast_vector<weight, rbm_t::num_hidden>> p_h_s;

    bool init = true;

    cd_trainer(rbm_t& rbm) : base_cd_trainer<rbm_t>(rbm), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    weight train_batch(const dll::batch<T>& batch){
        static etl::fast_matrix<weight, rbm_t::num_visible, rbm_t::num_hidden> t1;
        static etl::fast_matrix<weight, rbm_t::num_visible, rbm_t::num_hidden> t2;

        return train_normal<false, N>(batch, rbm, *this, t1, t2);
    }

    static std::string name(){
        return "Contrastive Divergence";
    }
};

/*!
 * \brief Contrastive divergence trainer for RBM.
 */
template<std::size_t N, typename RBM>
struct cd_trainer<N, RBM, enable_if_t<rbm_traits<RBM>::is_dynamic()>> : base_cd_trainer<RBM> {
    static_assert(N > 0, "CD-0 is not a valid training method");

    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    rbm_t& rbm;

    std::vector<etl::dyn_vector<weight>> p_h_a;
    std::vector<etl::dyn_vector<weight>> p_h_s;

    bool init = true;

    cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(rbm), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    weight train_batch(const dll::batch<T>& batch){
        static etl::dyn_matrix<weight> t1(rbm.num_visible, rbm.num_hidden);
        static etl::dyn_matrix<weight> t2(rbm.num_visible, rbm.num_hidden);

        return train_normal<false, N>(batch, rbm, *this, t1, t2);
    }

    static std::string name(){
        return "Contrastive Divergence";
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
    cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(rbm), rbm(rbm) {
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

    std::vector<etl::fast_vector<weight, rbm_t::num_hidden>> p_h_a;
    std::vector<etl::fast_vector<weight, rbm_t::num_hidden>> p_h_s;

    bool init = true;

    rbm_t& rbm;

    persistent_cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(rbm),
            p_h_a(rbm_traits<rbm_t>::batch_size()),
            p_h_s(rbm_traits<rbm_t>::batch_size()), rbm(rbm) {
        //Nothing else to init here
    }

    template<typename T>
    weight train_batch(const dll::batch<T>& batch){
        static etl::fast_matrix<weight, rbm_t::num_visible, rbm_t::num_hidden> t1;
        static etl::fast_matrix<weight, rbm_t::num_visible, rbm_t::num_hidden> t2;

        return train_normal<true, K>(batch, rbm, *this, t1, t2);
    }

    static std::string name(){
        return "Persistent Contrastive Divergence";
    }
};

/*!
 * \brief Persistent Contrastive Divergence Trainer for RBM.
 */
template<std::size_t K, typename RBM>
struct persistent_cd_trainer<K, RBM, enable_if_t<rbm_traits<RBM>::is_dynamic()>> : base_cd_trainer<RBM> {
    static_assert(K > 0, "PCD-0 is not a valid training method");

    typedef RBM rbm_t;
    typedef typename rbm_t::weight weight;

    std::vector<etl::dyn_vector<weight>> p_h_a;
    std::vector<etl::dyn_vector<weight>> p_h_s;

    bool init = true;

    rbm_t& rbm;

    persistent_cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(rbm), rbm(rbm) {
        for(std::size_t i = 0; i < get_batch_size(rbm); ++i){
            p_h_a.emplace_back(rbm.num_hidden);
            p_h_s.emplace_back(rbm.num_hidden);
        }
    }

    template<typename T>
    weight train_batch(const dll::batch<T>& batch){
        static etl::dyn_matrix<weight> t1(rbm.num_visible, rbm.num_hidden);
        static etl::dyn_matrix<weight> t2(rbm.num_visible, rbm.num_hidden);

        return train_normal<true, K>(batch, rbm, *this, t1, t2);
    }

    static std::string name(){
        return "Persistent Contrastive Divergence";
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
    persistent_cd_trainer(rbm_t& rbm) : base_cd_trainer<RBM>(rbm),
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