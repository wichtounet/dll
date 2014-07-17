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

template<typename DBN>
struct cg_trainer {
    using dbn_t = DBN;

    template<typename T, typename L>
    void train_batch(std::size_t, const dll::batch<T>&, const dll::batch<L>&, DBN&){
        //TODO :)
    }
};

} //end of dbn namespace

#endif