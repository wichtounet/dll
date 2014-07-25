//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*! \file Stochastic Gradient Descent (SGD) Implementation */

#ifndef DBN_STOCHASTIC_GRADIENT_DESCENT
#define DBN_STOCHASTIC_GRADIENT_DESCENT

namespace dll {

template<typename DBN, bool Debug = false>
struct sgd_trainer {
    using dbn_t = DBN;
    using weight = typename dbn_t::weight;

    static constexpr const std::size_t layers = dbn_t::layers;

    dbn_t& dbn;
    typename dbn_t::tuple_type& tuples;

    sgd_trainer(dbn_t& dbn) : dbn(dbn), tuples(dbn.tuples) {}

    void init_training(std::size_t batch_size){
        //TODO 
    }

    template<typename T, typename L>
    void train_batch(std::size_t epoch, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        //TODO 
    }
};

} //end of dbn namespace

#endif