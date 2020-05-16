//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Support for reshaping a sample into a batch-like structure
 * using etl::reshape
 */

#pragma once

#include "etl/etl.hpp"

namespace dll {

/*!
 * \brief Reshape the given expression into a batch (1 sample)
 * \param expr The expression to reshape
 * \return the reshaped expression
 */
template<typename Expr>
decltype(auto) batch_reshape(Expr&& expr){
    if constexpr (etl::is_fast<Expr>) {
        if constexpr (etl::dimensions<Expr>() == 1) {
            return etl::reshape<1, etl::dim<0, Expr>()>(expr);
        } else if constexpr (etl::dimensions<Expr>() == 2) {
            return etl::reshape<1, etl::dim<0, Expr>(), etl::dim<1, Expr>()>(expr);
        } else if constexpr (etl::dimensions<Expr>() == 3) {
            return etl::reshape<1, etl::dim<0, Expr>(), etl::dim<1, Expr>(), etl::dim<2, Expr>()>(expr);
        }
    } else {
        if constexpr (etl::dimensions<Expr>() == 1) {
            return etl::reshape(expr, 1, etl::dim<0>(expr));
        } else if constexpr (etl::dimensions<Expr>() == 2) {
            return etl::reshape(expr, 1, etl::dim<0>(expr), etl::dim<1>(expr));
        } else if constexpr (etl::dimensions<Expr>() == 3) {
            return etl::reshape(expr, 1, etl::dim<0>(expr), etl::dim<1>(expr), etl::dim<2>(expr));
        }
    }

    cpp_unreachable("Invalid selection in batch_reshape");
}

} //end of dll namespace
