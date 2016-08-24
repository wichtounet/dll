//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Trainer context forward declarations
 */

#pragma once

namespace dll {

/*!
 * \brief The context of a layer during SGD training
 * \tparam DBN The containing DBN
 * \tparam Layer The layer
 */
template <typename DBN, typename Layer, typename Enable = void>
struct sgd_context;

/*!
 * \brief The context of a RBM during CG training
 * \tparam RBM The RBM.
 */
template <typename RBM>
struct cg_context {};

} //end of dll namespace
