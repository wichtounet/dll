//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_RBM_BASE_HPP
#define DBN_RBM_BASE_HPP

namespace dll {

/*!
 * \brief Restricted Boltzmann Machine
 */
template<typename Layer>
class rbm_base {
public:
    typedef double weight;
    typedef double value_t;

    using conf = Layer;

    //Configurable properties
    weight learning_rate = 1e-1;

    weight initial_momentum = 0.5;
    weight final_momentum = 0.9;
    weight final_momentum_epoch = 6;

    weight momentum;

    weight weight_cost = 0.0002;

    weight sparsity_target = 0.01;
    weight decay_rate = 0.99;
    weight sparsity_cost = 1.0;

    //No copying
    rbm_base(const rbm_base& rbm) = delete;
    rbm_base& operator=(const rbm_base& rbm) = delete;

    //No moving
    rbm_base(rbm_base&& rbm) = delete;
    rbm_base& operator=(rbm_base&& rbm) = delete;

    rbm_base(){
        //Nothing to do
    }
};

} //end of dbn namespace

#endif