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
 * \brief Base class for Restricted Boltzmann Machine.
 *
 * It only contains configurable properties that are used by each
 * version of RBM.
 */
template<typename Layer>
class rbm_base {
public:
    typedef double weight;
    typedef double value_t;

    using conf = Layer;

    //Configurable properties
    weight learning_rate = 1e-1;        ///< The learning rate

    weight initial_momentum = 0.5;      ///< The initial momentum
    weight final_momentum = 0.9;        ///< The final momentum applied after *final_momentum_epoch* epoch
    weight final_momentum_epoch = 6;    ///< The epoch at which momentum change

    weight momentum = 0;                ///< The current momentum

    weight weight_cost = 0.0002;        ///< The weight cost for weight decay

    weight sparsity_target = 0.01;      ///< The sparsity target
    weight decay_rate = 0.99;           ///< The sparsity decay rate
    weight sparsity_cost = 1.0;         ///< The sparsity cost (or sparsity multiplier)

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