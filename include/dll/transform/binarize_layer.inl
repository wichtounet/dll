//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/transform/transform_layer.hpp"

namespace dll {

/*!
 * \brief Simple thresholding binarize layer
 *
 * Note: This is only supported at the beginning of the network, no
 * backpropagation is possible for now.
 */
template <typename Desc>
struct binarize_layer : transform_layer<binarize_layer<Desc>> {
    using desc      = Desc;                                  ///< The descriptor type
    using base_type = transform_layer<binarize_layer<Desc>>; ///< The base type

    static constexpr size_t Threshold = desc::T;

    binarize_layer() = default;

    /*!
     * \brief Returns a string representation of the layer
     */
    static std::string to_short_string() {
        return "Binarize";
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    static void forward_batch(Output& output, const Input& input) {
        output = input;

        for (auto& value : output) {
            value = value > Threshold ? 1 : 0;
        }
    }

    /*!
     * \brief Adapt the errors, called before backpropagation of the errors.
     *
     * This must be used by layers that have both an activation fnction and a non-linearity.
     *
     * \param context the training context
     */
    template<typename C>
    void adapt_errors(C& context) const {
        cpp_unused(context);
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        cpp_unused(output);
        cpp_unused(context);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        cpp_unused(context);
    }
};

//Allow odr-use of the constexpr static members

template <typename Desc>
const size_t binarize_layer<Desc>::Threshold;

// Declare the traits for the layer

template<typename Desc>
struct layer_base_traits<binarize_layer<Desc>> {
    static constexpr bool is_neural     = false; ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false; ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false; ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = true;  ///< Indicates if the layer is a transform layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

/*!
 * \brief Specialization of sgd_context for binarize_layer
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, binarize_layer<Desc>, L> {
    using layer_t          = binarize_layer<Desc>;                            ///< The current layer type
    using previous_layer   = typename DBN::template layer_type<L - 1>;          ///< The previous layer type
    using previous_context = sgd_context<DBN, previous_layer, L - 1>;           ///< The previous layer's context
    using inputs_t         = decltype(std::declval<previous_context>().output); ///< The type of inputs

    inputs_t input;  ///< A batch of input
    inputs_t output; ///< A batch of output
    inputs_t errors; ///< A batch of errors

    sgd_context(layer_t& /*layer*/){}
};

/*!
 * \brief Specialization of cg_context for binarize_layer
 */
template <typename Desc>
struct cg_context<binarize_layer<Desc>> {
    using rbm_t  = binarize_layer<Desc>;
    using weight = double; ///< The data type for this layer

    static constexpr bool is_trained = false;

    static constexpr size_t num_visible = 1;
    static constexpr size_t num_hidden  = 1;

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
