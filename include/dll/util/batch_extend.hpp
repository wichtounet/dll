//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Support for extending a sample into a batch-like structure
 */

#pragma once

#include "etl/etl.hpp"

namespace dll {

/*!
 * \brief Extend the given one expression into a batch of the same
 * size as the given batch
 *
 * \param batch The batch expression
 * \param one The one expression
 *
 * \return the extended batchession
 */
template<typename Batch, typename One, cpp_enable_if(etl::all_fast<Batch, One>::value && etl::dimensions<One>() == 1)>
decltype(auto) batch_extend(Batch&& batch, One&& one){
    cpp_unused(batch);
    cpp_unused(one);
    return etl::fast_dyn_matrix<etl::value_t<One>, etl::dim<0, Batch>(), etl::dim<0, One>()>();
}

/*!
 * \brief Extend the given one expression into a batch of the same
 * size as the given batch
 *
 * \param batch The batch expression
 * \param one The one expression
 *
 * \return the extended batchession
 */
template<typename Batch, typename One, cpp_enable_if(etl::all_fast<Batch, One>::value && etl::dimensions<One>() == 2)>
decltype(auto) batch_extend(Batch&& batch, One&& one){
    cpp_unused(batch);
    cpp_unused(one);
    return etl::fast_dyn_matrix<etl::value_t<One>, etl::dim<0, Batch>(), etl::dim<0, One>(), etl::dim<1, One>()>();
}

/*!
 * \brief Extend the given one expression into a batch of the same
 * size as the given batch
 *
 * \param batch The batch expression
 * \param one The one expression
 *
 * \return the extended batchession
 */
template<typename Batch, typename One, cpp_enable_if(etl::all_fast<Batch, One>::value && etl::dimensions<One>() == 3)>
decltype(auto) batch_extend(Batch&& batch, One&& one){
    cpp_unused(batch);
    cpp_unused(one);
    return etl::fast_dyn_matrix<etl::value_t<One>, etl::dim<0, Batch>(), etl::dim<0, One>(), etl::dim<1, One>(), etl::dim<2, One>()>();
}

/*!
 * \brief Extend the given one expression into a batch of the same
 * size as the given batch
 *
 * \param batch The batch expression
 * \param one The one expression
 *
 * \return the extended batchession
 */
template<typename Batch, typename One, cpp_enable_if(!etl::all_fast<Batch, One>::value && etl::dimensions<One>() == 1)>
decltype(auto) batch_extend(Batch&& batch, One&& one){
    return etl::dyn_matrix<etl::value_t<One>, etl::dimensions<One>() + 1>(etl::dim<0>(batch), etl::dim<0>(one));
}

/*!
 * \brief Extend the given one expression into a batch of the same
 * size as the given batch
 *
 * \param batch The batch expression
 * \param one The one expression
 *
 * \return the extended batchession
 */
template<typename Batch, typename One, cpp_enable_if(!etl::all_fast<Batch, One>::value && etl::dimensions<One>() == 2)>
decltype(auto) batch_extend(Batch&& batch, One&& one){
    return etl::dyn_matrix<etl::value_t<One>, etl::dimensions<One>() + 1>(etl::dim<0>(batch), etl::dim<0>(one), etl::dim<1>(one));
}

/*!
 * \brief Extend the given one expression into a batch of the same
 * size as the given batch
 *
 * \param batch The batch expression
 * \param one The one expression
 *
 * \return the extended batchession
 */
template<typename Batch, typename One, cpp_enable_if(!etl::all_fast<Batch, One>::value && etl::dimensions<One>() == 3)>
decltype(auto) batch_extend(Batch&& batch, One&& one){
    return etl::dyn_matrix<etl::value_t<One>, etl::dimensions<One>() + 1>(etl::dim<0>(batch), etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));
}

} //end of dll namespace
