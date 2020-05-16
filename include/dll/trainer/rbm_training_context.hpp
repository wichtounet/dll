//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief A container for information collected during training of a RBM.
 *
 * The values contained in this structure are only valid at the end of the epoch
 * training, intermediate values can be contained during intermediate batch
 * training.
 */
struct rbm_training_context {
    double reconstruction_error = 0.0; ///< The mean reconstruction error
    double free_energy          = 0.0; ///< The mean free energy
    double sparsity             = 0.0; ///< The mean sparsity

    double batch_error    = 0.0; ///< The mean reconstruction error for the last batch
    double batch_sparsity = 0.0; ///< The mean sparsity for the last batch
};

} //end of dll namespace
