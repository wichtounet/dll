//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*! \file Conjugate Gradient (CG) descent Implementation */

#ifndef DBN_CONJUGATE_GRADIENT_HPP
#define DBN_CONJUGATE_GRADIENT_HPP

namespace dll {

//TODO Remove later
template<typename Target>
struct gradient_context;

template<typename DBN>
struct cg_trainer {
    using dbn_t = DBN;

    dbn_t& dbn;

    cg_trainer(dbn_t& dbn) : dbn(dbn) {}

    void init_training(std::size_t batch_size){
        detail::for_each(dbn.tuples, [batch_size](auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            constexpr const auto num_hidden = rbm_t::num_hidden;

            for(size_t i = 0; i < batch_size; ++i){
                rbm.gr_probs_a.emplace_back(num_hidden);
                rbm.gr_probs_s.emplace_back(num_hidden);
            }
        });
    }

    template<typename T, typename L>
    void train_batch(std::size_t epoch, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        gradient_context<L> context(data_batch, label_batch, epoch);

        dbn.minimize(context);
    }
};

} //end of dbn namespace

#endif