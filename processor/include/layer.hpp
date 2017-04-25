//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <string>
#include <vector>
#include <iostream>

#include "dll/processor/processor.hpp"

namespace dllp {

struct layer;

using layers_t = std::vector<std::unique_ptr<dllp::layer>>;

/*!
 * \brief A layer (in the processor configuration)
 */
struct layer {
    /*!
     * \brief Print the layer code to the given stream
     * \param out The stream to print to
     */
    virtual void print(std::ostream& out) const = 0;

    /*!
     * \brief Returns the number of hidden unit
     */
    virtual std::size_t hidden_get() const = 0;

    /*!
     * \brief Indicates if the layer is convolutional
     */
    virtual bool is_conv() const {
        return false;
    }

    /*!
     * \brief Returns the first dimension of the output matrix
     */
    virtual std::size_t hidden_get_1() const {
        return 0;
    }

    /*!
     * \brief Returns the second dimension of the output matrix
     */
    virtual std::size_t hidden_get_2() const {
        return 0;
    }

    /*!
     * \brief Returns the third dimension of the output matrix
     */
    virtual std::size_t hidden_get_3() const {
        return 0;
    }

    virtual bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) = 0;

    virtual void set(std::ostream& /*out*/, const std::string& /*lhs*/) const {/* Nothing */};
};

enum class parse_result {
    PARSED,
    ERROR,
    NOT_PARSED
};

struct base_rbm_layer : layer {
    std::string visible_unit; ///< The visible unit type
    std::string hidden_unit;  ///< The hidden unit type

    double learning_rate   = dll::processor::stupid_default; ///< The learning rate
    double momentum        = dll::processor::stupid_default; ///< The momentum
    std::size_t batch_size = 0;                              ///< The batch size

    std::string decay     = "none";                         ///< The type of decay
    double l1_weight_cost = dll::processor::stupid_default; ///< The L1 decay rate
    double l2_weight_cost = dll::processor::stupid_default; ///< The L2 decay rate

    std::string sparsity   = "none";
    double sparsity_target = dll::processor::stupid_default;
    double pbias_lambda    = dll::processor::stupid_default;
    double pbias           = dll::processor::stupid_default;

    std::string trainer = "cd";

    bool parallel_mode = false;
    bool shuffle       = false;

    void print(std::ostream& out) const override;
    void set(std::ostream& out, const std::string& lhs) const override;

    parse_result base_parse(const std::vector<std::string>& lines, std::size_t& i);
};

struct rbm_layer : base_rbm_layer {
    std::size_t visible = 0; ///< The number of visible units
    std::size_t hidden  = 0; ///< The number of hidden units

    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;

    std::size_t hidden_get() const override;
};

struct conv_rbm_layer : base_rbm_layer {
    std::size_t c  = 0; ///< The number of channels
    std::size_t v1 = 0; ///< The first dimension of the output
    std::size_t v2 = 0; ///< The second dimension of the output
    std::size_t k  = 0; ///< The number of filters
    std::size_t w1 = 0; ///< The first dimension of the filters
    std::size_t w2 = 0; ///< The second dimension of the filters

    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;

    bool is_conv() const override;
    std::size_t hidden_get() const override;
    std::size_t hidden_get_1() const override;
    std::size_t hidden_get_2() const override;
    std::size_t hidden_get_3() const override;
};

struct conv_rbm_mp_layer : base_rbm_layer {
    std::size_t c  = 0; ///< The number of channels
    std::size_t v1 = 0; ///< The first dimension of the output
    std::size_t v2 = 0; ///< The second dimension of the output
    std::size_t k  = 0; ///< The number of filters
    std::size_t w1 = 0; ///< The first dimension of the filters
    std::size_t w2 = 0; ///< The second dimension of the filters
    std::size_t p  = 0;

    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;

    bool is_conv() const override;
    std::size_t hidden_get() const override;
    std::size_t hidden_get_1() const override;
    std::size_t hidden_get_2() const override;
    std::size_t hidden_get_3() const override;
};

struct dense_layer : layer {
    std::size_t visible = 0; ///< The number of visible units
    std::size_t hidden  = 0; ///< The number of hidden units

    std::string activation;

    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;

    std::size_t hidden_get() const override;
};

struct conv_layer : layer {
    std::size_t c  = 0; ///< The number of channels
    std::size_t v1 = 0; ///< The first dimension of the input
    std::size_t v2 = 0; ///< The second dimension of the input
    std::size_t k  = 0; ///< The number of filters
    std::size_t w1 = 0; ///< The first dimension of the filters
    std::size_t w2 = 0; ///< The second dimension of the filters

    std::string activation;

    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;

    bool is_conv() const override;
    std::size_t hidden_get() const override;
    std::size_t hidden_get_1() const override;
    std::size_t hidden_get_2() const override;
    std::size_t hidden_get_3() const override;
};

struct pooling_layer : layer {
    std::size_t c  = 0; ///< The number of channels
    std::size_t v1 = 0; ///< The first dimension of the input
    std::size_t v2 = 0; ///< The second dimension of the input
    std::size_t c1 = 0; ///< The pooling factor of the first dimension
    std::size_t c2 = 0; ///< The pooling factor of the first dimension
    std::size_t c3 = 0; ///< The pooling factor of the first dimension

    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;

    bool is_conv() const override;
    std::size_t hidden_get() const override;
    std::size_t hidden_get_1() const override;
    std::size_t hidden_get_2() const override;
    std::size_t hidden_get_3() const override;
};

struct mp_layer : pooling_layer {
    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;
};

struct avgp_layer : pooling_layer {
    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;
};

} //end of namespace dllp
