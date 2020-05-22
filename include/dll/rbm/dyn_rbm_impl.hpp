//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/etl.hpp"

#include "dll/base_traits.hpp"
#include "dll/rbm/standard_rbm.hpp"

namespace dll {

/*!
 * \brief Standard version of Restricted Boltzmann Machine
 *
 * This follows the definition of a RBM by Geoffrey Hinton.
 */
template <typename Desc>
struct dyn_rbm_impl final : public standard_rbm<dyn_rbm_impl<Desc>, Desc> {
    using desc      = Desc; ///< The descriptor of the layer
    using weight    = typename desc::weight; ///< The data type for this layer
    using this_type = dyn_rbm_impl<Desc>; ///< The type of this layer
    using base_type = standard_rbm<this_type, Desc>;
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    using input_t      = typename rbm_base_traits<this_type>::input_t; ///< The type of the input
    using output_t     = typename rbm_base_traits<this_type>::output_t; ///< The type of the output
    using input_one_t  = typename rbm_base_traits<this_type>::input_one_t; ///< The type of one input
    using output_one_t = typename rbm_base_traits<this_type>::output_one_t; ///< The type of one output

    static constexpr unit_type visible_unit = desc::visible_unit;
    static constexpr unit_type hidden_unit  = desc::hidden_unit;
    static constexpr size_t batch_size      = desc::BatchSize; ///< The mini-batch size

    using w_type = etl::dyn_matrix<weight>; ///< The type of the weights
    using b_type = etl::dyn_vector<weight>; ///< The type of the biases
    using c_type = etl::dyn_vector<weight>;

    //Weights and biases
    w_type w; ///< Weights
    b_type b; ///< Hidden biases
    c_type c; ///< Visible biases

    //Backup weights and biases
    std::unique_ptr<w_type> bak_w; ///< Backup Weights
    std::unique_ptr<b_type> bak_b; ///< Backup Hidden biases
    std::unique_ptr<c_type> bak_c; ///< Backup Visible biases

    //Reconstruction data
    etl::dyn_vector<weight> v1; ///< State of the visible units

    etl::dyn_vector<weight> h1_a; ///< Activation probabilities of hidden units after first CD-step
    etl::dyn_vector<weight> h1_s; ///< Sampled value of hidden units after first CD-step

    etl::dyn_vector<weight> v2_a; ///< Activation probabilities of visible units after first CD-step
    etl::dyn_vector<weight> v2_s; ///< Sampled value of visible units after first CD-step

    etl::dyn_vector<weight> h2_a; ///< Activation probabilities of hidden units after last CD-step
    etl::dyn_vector<weight> h2_s; ///< Sampled value of hidden units after last CD-step

    size_t num_visible;
    size_t num_hidden;

    dyn_rbm_impl() : base_type() {}

    /*!
     * \brief Initialize a RBM with basic weights.
     *
     * The weights are initialized from a normal distribution of
     * zero-mean and 0.1 variance.
     */
    dyn_rbm_impl(size_t num_visible, size_t num_hidden)
            : base_type(),
              w(num_visible, num_hidden),
              b(num_hidden, static_cast<weight>(0.0)),
              c(num_visible, static_cast<weight>(0.0)),
              v1(num_visible),
              h1_a(num_hidden),
              h1_s(num_hidden),
              v2_a(num_visible),
              v2_s(num_visible),
              h2_a(num_hidden),
              h2_s(num_hidden),
              num_visible(num_visible),
              num_hidden(num_hidden) {
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        w = etl::normal_generator<weight>() * 0.1;
    }

    /*!
     * \brief Initialize the dynamic layer
     */
    void init_layer(size_t nv, size_t nh) {
        num_visible = nv;
        num_hidden  = nh;

        w    = etl::dyn_matrix<weight>(num_visible, num_hidden);
        b    = etl::dyn_vector<weight>(num_hidden, static_cast<weight>(0.0));
        c    = etl::dyn_vector<weight>(num_visible, static_cast<weight>(0.0));
        v1   = etl::dyn_vector<weight>(num_visible);
        h1_a = etl::dyn_vector<weight>(num_hidden);
        h1_s = etl::dyn_vector<weight>(num_hidden);
        v2_a = etl::dyn_vector<weight>(num_visible);
        v2_s = etl::dyn_vector<weight>(num_visible);
        h2_a = etl::dyn_vector<weight>(num_hidden);
        h2_s = etl::dyn_vector<weight>(num_hidden);

        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        w = etl::normal_generator<weight>() * 0.1;
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    size_t input_size() const noexcept {
        return num_visible;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    size_t output_size() const noexcept {
        return num_hidden;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    size_t parameters() const noexcept {
        return num_visible * num_hidden;
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_short_string([[maybe_unused]] std::string pre = "") const {
        char buffer[1024];
        snprintf(
            buffer, 1024, "RBM(%s) (dyn)",
            to_string(hidden_unit).c_str());
        return {buffer};
    }

    /*!
     * \brief Returns a short description of the layer
     * \return an std::string containing a short description of the layer
     */
    std::string to_full_string([[maybe_unused]] std::string pre = "") const {
        char buffer[1024];
        snprintf(
            buffer, 1024, "RBM(dyn)(%s): %lu -> %lu",
            to_string(hidden_unit).c_str(), num_visible, num_hidden);
        return {buffer};
    }

    /*!
     * \brief Returns the output shape
     * \return an std::string containing the description of the output shape
     */
    std::vector<size_t> output_shape([[maybe_unused]] const std::vector<size_t>& input_shape) const {
        return {num_hidden};
    }

    /*!
     * \brief Apply the layer to the batch of input
     * \param output The batch of output
     * \param input The batch of input to apply the layer to
     */
    template <typename Input, typename Output>
    void forward_batch(Output& output, const Input& input) const {
        this->batch_activate_hidden(output, input);
    }

    // This is specific to dyn because of the nv/nh
    void init_cg_context() {
        if (!this->cg_context_ptr) {
            this->cg_context_ptr = std::make_shared<cg_context<this_type>>(num_visible, num_hidden);
        }
    }

    void prepare_input(input_one_t& input) const {
        input = input_one_t(num_visible);
    }

    /*!
     * \brief Initialize the dynamic version of the layer from the
     * fast version of the layer
     * \param dyn Reference to the dynamic version of the layer that
     * needs to be initialized
     */
    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
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
        static_assert(
            hidden_unit == unit_type::BINARY || hidden_unit == unit_type::RELU || hidden_unit == unit_type::SOFTMAX,
            "Only (C)RBM with binary, softmax or RELU hidden unit are supported");

        static constexpr function activation_function =
            hidden_unit == unit_type::BINARY
                ? function::SIGMOID
                : (hidden_unit == unit_type::SOFTMAX ? function::SOFTMAX : function::RELU);

        context.errors = f_derivative<activation_function>(context.output) >> context.errors;
    }

    /*!
     * \brief Backpropagate the errors to the previous layers
     * \param output The ETL expression into which write the output
     * \param context The training context
     */
    template<typename H, typename C>
    void backward_batch(H&& output, C& context) const {
        // The reshape has no overhead, so better than SFINAE for nothing
        const auto Batch = etl::dim<0>(output);
        etl::reshape(output, Batch, num_visible) = context.errors * etl::transpose(w);
    }

    /*!
     * \brief Compute the gradients for this layer, if any
     * \param context The trainng context
     */
    template<typename C>
    void compute_gradients(C& context) const {
        std::get<0>(context.up.context)->grad = batch_outer(context.input, context.errors);
        std::get<1>(context.up.context)->grad = bias_batch_sum_2d(context.errors);
    }
};

/*!
 * \brief Simple traits to pass information around from the real
 * class to the CRTP class.
 */
template <typename Desc>
struct rbm_base_traits<dyn_rbm_impl<Desc>> {
    using desc      = Desc; ///< The descriptor of the layer
    using weight    = typename desc::weight; ///< The data type for this layer

    using input_one_t  = etl::dyn_vector<weight>; ///< The type of one input
    using output_one_t = etl::dyn_vector<weight>; ///< The type of one output
    using input_t      = std::vector<input_one_t>; ///< The type of the input
    using output_t     = std::vector<output_one_t>; ///< The type of the output
};

// Declare the traits for the RBM

template<typename Desc>
struct layer_base_traits<dyn_rbm_impl<Desc>> {
    static constexpr bool is_neural     = true;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = true;  ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = false; ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = true;  ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_recurrent  = false; ///< Indicates if the layer is a recurrent layer
    static constexpr bool is_multi      = false; ///< Indicates if the layer is a multi-layer layer
    static constexpr bool is_dynamic    = true;  ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = Desc::hidden_unit != dll::unit_type::SOFTMAX; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

template<typename Desc>
struct rbm_layer_base_traits<dyn_rbm_impl<Desc>> {
    using param = typename Desc::parameters;

    static constexpr bool has_momentum       = param::template contains<momentum>();                       ///< Does the RBM has momentum
    static constexpr bool has_clip_gradients = param::template contains<clip_gradients>();                 ///< Does the RBM has gradient clipping
    static constexpr bool is_verbose         = param::template contains<verbose>();                        ///< Does the RBM is verbose
    static constexpr bool has_shuffle        = param::template contains<shuffle>();                        ///< Does the RBM has shuffle
    static constexpr bool is_dbn_only        = param::template contains<dbn_only>();                       ///< Does the RBM is only used inside a DBN
    static constexpr bool has_init_weights   = param::template contains<init_weights>();                   ///< Does the RBM use weights initialization
    static constexpr bool has_free_energy    = param::template contains<free_energy>();                    ///< Does the RBM displays the free energy
    static constexpr auto sparsity_method    = get_value_l_v<sparsity<dll::sparsity_method::NONE>, param>; ///< The RBM's sparsity method
    static constexpr auto bias_mode          = get_value_l_v<bias<dll::bias_mode::NONE>, param>;           ///< The RBM's sparsity bias mode
    static constexpr auto decay              = get_value_l_v<weight_decay<dll::decay_type::NONE>, param>;  ///< The RMB's sparsity decay type
    static constexpr bool has_sparsity       = sparsity_method != dll::sparsity_method::NONE;              ///< Does the RBM has sparsity
};

/*!
 * \brief Specialization of sgd_context for dyn_rbm_impl
 */
template <typename DBN, typename Desc, size_t L>
struct sgd_context<DBN, dyn_rbm_impl<Desc>, L> {
    using layer_t = dyn_rbm_impl<Desc>;
    using weight  = typename layer_t::weight; ///< The data type for this layer

    static constexpr auto batch_size = DBN::batch_size;

    etl::dyn_matrix<weight, 2> input;
    etl::dyn_matrix<weight, 2> output;
    etl::dyn_matrix<weight, 2> errors;

    sgd_context(const layer_t& layer)
            : input(batch_size, layer.num_visible, 0.0), output(batch_size, layer.num_hidden, 0.0), errors(batch_size, layer.num_hidden, 0.0) {}
};

/*!
 * \brief Specialzation of cg_context for dyn_rbm_impl
 */
template <typename Desc>
struct cg_context<dyn_rbm_impl<Desc>> {
    using rbm_t  = dyn_rbm_impl<Desc>;
    using weight = typename rbm_t::weight; ///< The data type for this layer

    static constexpr bool is_trained = true;

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

    cg_context(size_t num_visible, size_t num_hidden) :
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

} //end of dll namespace
