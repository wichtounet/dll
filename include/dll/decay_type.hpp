//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief Define how weight decay is applied.
 */
enum class decay_type {
    NONE,     ///< No weight decay is applied during training
    L1,       ///< Apply L1 weight decay on weights
    L1_FULL,  ///< Apply L1 weight decay on weights and biases
    L2,       ///< Apply L2 weight decay on weights
    L2_FULL,  ///< Apply L2 weight decay on weights and biases
    L1L2,     ///< Apply L1/L2 weight decay on weights
    L1L2_FULL ///< Apply L1/L2 weight decay on weights and biases
};

/*!
 * \brief Indicates the type of decay that is to be applied to weights
 * \param t The RBM weight decay type.
 * \return one of L1,L2,NONE
 */
constexpr decay_type w_decay(decay_type t) {
    return
        (t == decay_type::L1 || t == decay_type::L1_FULL)     ? decay_type::L1 :
        (t == decay_type::L2 || t == decay_type::L2_FULL)     ? decay_type::L2 :
        (t == decay_type::L1L2 || t == decay_type::L1L2_FULL) ? decay_type::L1L2
                                                              : decay_type::NONE;
}

/*!
 * \brief Indicates the type of decay that is to be applied to biases
 * \param t The RBM weight decay type.
 * \return one of L1,L2,NONE
 */
constexpr decay_type b_decay(decay_type t) {
    return t == decay_type::L1_FULL ? decay_type::L1 : t == decay_type::L2_FULL ? decay_type::L2 : t == decay_type::L1L2_FULL ? decay_type::L1L2 : decay_type::NONE;
}

} //end of dll namespace
