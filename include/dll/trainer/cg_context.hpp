//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file cg_context.hpp
 * \brief Conjugate Gradient (CG) descent context implementation.
 */

#pragma once

#include "etl/etl.hpp"

namespace dll {

//Foward
template <typename Desc>
struct rbm;

/*!
 * \copydoc cg_context
 */
template <typename Desc>
struct cg_context<rbm<Desc>> {
    using rbm_t  = rbm<Desc>;
    using weight = typename rbm_t::weight;

    static constexpr const bool is_trained = true;

    static constexpr const std::size_t num_visible = rbm_t::num_visible;
    static constexpr const std::size_t num_hidden  = rbm_t::num_hidden;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_incs;
    etl::fast_vector<weight, num_hidden> gr_b_incs;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_best;
    etl::fast_vector<weight, num_hidden> gr_b_best;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_best_incs;
    etl::fast_vector<weight, num_hidden> gr_b_best_incs;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_df0;
    etl::fast_vector<weight, num_hidden> gr_b_df0;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_df3;
    etl::fast_vector<weight, num_hidden> gr_b_df3;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_s;
    etl::fast_vector<weight, num_hidden> gr_b_s;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_tmp;
    etl::fast_vector<weight, num_hidden> gr_b_tmp;

    std::vector<etl::dyn_vector<weight>> gr_probs_a;
    std::vector<etl::dyn_vector<weight>> gr_probs_s;
};

//Foward
template <typename Desc>
struct dyn_rbm;

/*!
 * \copydoc cg_context
 */
template <typename Desc>
struct cg_context<dyn_rbm<Desc>> {
    using rbm_t  = dyn_rbm<Desc>;
    using weight = typename rbm_t::weight;

    static constexpr const bool is_trained = true;

    etl::dyn_matrix<weight, 2> gr_w_incs;
    etl::dyn_matrix<weight, 1> gr_b_incs;

    etl::dyn_matrix<weight, 2> gr_w_best;
    etl::dyn_matrix<weight, 1> gr_b_best;

    etl::dyn_matrix<weight, 2> gr_w_best_incs;
    etl::dyn_matrix<weight, 1> gr_b_best_incs;

    etl::dyn_matrix<weight, 2> gr_w_df0;
    etl::dyn_matrix<weight, 1> gr_b_df0;

    etl::dyn_matrix<weight, 2> gr_w_df3;
    etl::dyn_matrix<weight, 1> gr_b_df3;

    etl::dyn_matrix<weight, 2> gr_w_s;
    etl::dyn_matrix<weight, 1> gr_b_s;

    etl::dyn_matrix<weight, 2> gr_w_tmp;
    etl::dyn_matrix<weight, 1> gr_b_tmp;

    std::vector<etl::dyn_vector<weight>> gr_probs_a;
    std::vector<etl::dyn_vector<weight>> gr_probs_s;

    cg_context(std::size_t num_visible, std::size_t num_hidden) :
        gr_w_incs(num_visible, num_hidden), gr_b_incs(num_hidden),
        gr_w_best(num_visible, num_hidden), gr_b_best(num_hidden),
        gr_w_best_incs(num_visible, num_hidden), gr_b_best_incs(num_hidden),
        gr_w_df0(num_visible, num_hidden), gr_b_df0(num_hidden),
        gr_w_df3(num_visible, num_hidden), gr_b_df3(num_hidden),
        gr_w_s(num_visible, num_hidden), gr_b_s(num_hidden),
        gr_w_tmp(num_visible, num_hidden), gr_b_tmp(num_hidden)
    {
        //Nothing else to init
    }
};

//Forward
template <typename Desc>
struct binarize_layer;

/*!
 * \copydoc cg_context
 */
template <typename Desc>
struct cg_context<binarize_layer<Desc>> {
    using rbm_t  = binarize_layer<Desc>;
    using weight = double;

    static constexpr const bool is_trained = false;

    static constexpr const std::size_t num_visible = 1;
    static constexpr const std::size_t num_hidden  = 1;

    etl::fast_matrix<weight, 1, 1> gr_w_incs;
    etl::fast_vector<weight, 1> gr_b_incs;

    etl::fast_matrix<weight, 1, 1> gr_w_best;
    etl::fast_vector<weight, 1> gr_b_best;

    etl::fast_matrix<weight, 1, 1> gr_w_best_incs;
    etl::fast_vector<weight, 1> gr_b_best_incs;

    etl::fast_matrix<weight, 1, 1> gr_w_df0;
    etl::fast_vector<weight, 1> gr_b_df0;

    etl::fast_matrix<weight, 1, 1> gr_w_df3;
    etl::fast_vector<weight, 1> gr_b_df3;

    etl::fast_matrix<weight, 1, 1> gr_w_s;
    etl::fast_vector<weight, 1> gr_b_s;

    etl::fast_matrix<weight, 1, 1> gr_w_tmp;
    etl::fast_vector<weight, 1> gr_b_tmp;

    std::vector<etl::dyn_vector<weight>> gr_probs_a;
    std::vector<etl::dyn_vector<weight>> gr_probs_s;
};

} //end of dll namespace
