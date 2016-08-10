//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file sgd_context.hpp
 * \brief Stochastic Gradient Descent (SGD) context Implementation.
 */

#pragma once

#include "dll/layer_traits.hpp"
#include "dll/dbn_traits.hpp"

namespace dll {

/*!
 * \brief The context of a layer during SGD training
 * \tparam DBN The containing DBN
 * \tparam Layer The layer
 */
template <typename DBN, typename Layer, typename Enable = void>
struct sgd_context;

/*!
 * \copydoc sgd_context
 */
template <typename DBN, typename Layer>
struct sgd_context<DBN, Layer, std::enable_if_t<layer_traits<Layer>::is_dense_layer() && !layer_traits<Layer>::is_dynamic()>> {
    using layer_t = Layer;
    using weight  = typename layer_t::weight;

    static constexpr const auto num_visible = layer_t::num_visible;
    static constexpr const auto num_hidden  = layer_t::num_hidden;

    static constexpr const auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, num_visible, num_hidden> w_grad;
    etl::fast_matrix<weight, num_hidden> b_grad;

    etl::fast_matrix<weight, num_visible, num_hidden> w_inc;
    etl::fast_matrix<weight, num_hidden> b_inc;

    etl::fast_matrix<weight, batch_size, num_hidden> output;
    etl::fast_matrix<weight, batch_size, num_hidden> errors;

    sgd_context()
            : w_inc(0.0), b_inc(0.0), output(0.0), errors(0.0) {}
};

/*!
 * \copydoc sgd_context
 */
template <typename DBN, typename Layer>
struct sgd_context<DBN, Layer, std::enable_if_t<layer_traits<Layer>::is_dense_layer() && layer_traits<Layer>::is_dynamic()>> {
    using layer_t = Layer;
    using weight  = typename layer_t::weight;

    static constexpr const auto batch_size = DBN::batch_size;

    etl::dyn_matrix<weight, 2> w_grad;
    etl::dyn_matrix<weight, 1> b_grad;

    etl::dyn_matrix<weight, 2> w_inc;
    etl::dyn_matrix<weight, 1> b_inc;

    etl::dyn_matrix<weight, 2> output;
    etl::dyn_matrix<weight, 2> errors;

    sgd_context(std::size_t num_visible, std::size_t num_hidden)
            : w_grad(num_visible, num_hidden), b_grad(num_hidden),
              w_inc(num_visible, num_hidden, 0.0), b_inc(num_hidden, 0.0),
              output(batch_size, num_hidden, 0.0), errors(batch_size, num_hidden, 0.0) {}
};

/*!
 * \copydoc sgd_context
 */
template <typename DBN, typename Layer>
struct sgd_context<DBN, Layer, std::enable_if_t<layer_traits<Layer>::is_convolutional_layer()>> {
    using layer_t = Layer;
    using weight  = typename layer_t::weight;

    static_assert(!layer_traits<layer_t>::has_probabilistic_max_pooling(), "Probabilistic Max Pooling is not supported in backpropagation");

    static constexpr const std::size_t NV1 = layer_t::NV1;
    static constexpr const std::size_t NV2 = layer_t::NV2;
    static constexpr const std::size_t NH1 = layer_t::NH1;
    static constexpr const std::size_t NH2 = layer_t::NH2;
    static constexpr const std::size_t NW1 = layer_t::NW1;
    static constexpr const std::size_t NW2 = layer_t::NW2;
    static constexpr const std::size_t NC  = layer_t::NC;
    static constexpr const std::size_t K   = layer_t::K;

    static constexpr const auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, NC, K, NW1, NW2> w_grad;
    etl::fast_matrix<weight, K> b_grad;

    etl::fast_matrix<weight, NC, K, NW1, NW2> w_inc;
    etl::fast_matrix<weight, K> b_inc;

    etl::fast_matrix<weight, batch_size, K, NH1, NH2> output;
    etl::fast_matrix<weight, batch_size, K, NH1, NH2> errors;

    sgd_context()
            : w_inc(0.0), b_inc(0.0), output(0.0), errors(0.0) {}
};

/*!
 * \copydoc sgd_context
 */
template <typename DBN, typename Layer>
struct sgd_context<DBN, Layer, std::enable_if_t<layer_traits<Layer>::is_pooling_layer()>> {
    using layer_t = Layer;
    using weight  = typename layer_t::weight;

    static constexpr const std::size_t I1 = layer_t::I1;
    static constexpr const std::size_t I2 = layer_t::I2;
    static constexpr const std::size_t I3 = layer_t::I3;

    static constexpr const std::size_t O1 = layer_t::O1;
    static constexpr const std::size_t O2 = layer_t::O2;
    static constexpr const std::size_t O3 = layer_t::O3;

    static constexpr const auto batch_size = DBN::batch_size;

    etl::fast_matrix<weight, batch_size, I1, I2, I3> input;
    etl::fast_matrix<weight, batch_size, O1, O2, O3> output;
    etl::fast_matrix<weight, batch_size, O1, O2, O3> errors;
};

// This selector is used to compute the output types of transform layers
// Ideally, it should always be set to dyn_matrix, unfortunately this is not
// supported by the different rbm implementations who are waiting for fast
// matrix

template <typename DBN, typename Layer, typename Enable = void>
struct transform_output_type;

template <typename DBN, typename Layer>
struct transform_output_type <DBN, Layer, std::enable_if_t<dbn_traits<DBN>::is_dynamic()>> {
    static constexpr const auto dimensions = dbn_traits<DBN>::is_convolutional() ? 4 : 2;

    using weight  = typename DBN::weight;
    using type = etl::dyn_matrix<weight, dimensions>;
};

template <typename DBN, typename Layer>
struct transform_output_type <DBN, Layer, std::enable_if_t<!dbn_traits<DBN>::is_dynamic()>> {
    static constexpr const auto batch_size = DBN::batch_size;

    using type = typename DBN::template input_batch_t<batch_size>;
};

template <typename DBN, typename Layer>
using transform_output_type_t = typename transform_output_type<DBN, Layer>::type;

/*!
 * \copydoc sgd_context
 */
template <typename DBN, typename Layer>
struct sgd_context<DBN, Layer, std::enable_if_t<layer_traits<Layer>::is_transform_layer()>> {
    using layer_t = Layer;
    using weight  = typename DBN::weight;

    using inputs_t = transform_output_type_t<DBN, Layer>;

    inputs_t output;
    inputs_t errors;
};

} //end of dll namespace
