//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

struct layer {
    virtual void print(std::ostream& out) const = 0;
    virtual std::size_t hidden_get() const = 0;

    virtual bool is_conv() const {
        return false;
    }

    virtual std::size_t hidden_get_1() const {
        return 0;
    }
    virtual std::size_t hidden_get_2() const {
        return 0;
    }
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
    std::string visible_unit;
    std::string hidden_unit;

    double learning_rate   = dll::processor::stupid_default;
    double momentum        = dll::processor::stupid_default;
    std::size_t batch_size = 0;

    std::string decay     = "none";
    double l1_weight_cost = dll::processor::stupid_default;
    double l2_weight_cost = dll::processor::stupid_default;

    std::string sparsity   = "none";
    double sparsity_target = dll::processor::stupid_default;

    std::string trainer = "cd";

    bool parallel_mode = false;
    bool shuffle       = false;

    void print(std::ostream& out) const override;
    void set(std::ostream& out, const std::string& lhs) const override;

    parse_result base_parse(const std::vector<std::string>& lines, std::size_t& i);
};

struct rbm_layer : base_rbm_layer {
    std::size_t visible = 0;
    std::size_t hidden  = 0;

    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;

    std::size_t hidden_get() const override;
};

struct conv_rbm_layer : base_rbm_layer {
    std::size_t c  = 0;
    std::size_t v1 = 0;
    std::size_t v2 = 0;
    std::size_t k  = 0;
    std::size_t w1 = 0;
    std::size_t w2 = 0;

    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;

    bool is_conv() const override;
    std::size_t hidden_get() const override;
    std::size_t hidden_get_1() const override;
    std::size_t hidden_get_2() const override;
    std::size_t hidden_get_3() const override;
};

struct conv_rbm_mp_layer : base_rbm_layer {
    std::size_t c  = 0;
    std::size_t v1 = 0;
    std::size_t v2 = 0;
    std::size_t k  = 0;
    std::size_t w1 = 0;
    std::size_t w2 = 0;
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
    std::size_t visible = 0;
    std::size_t hidden  = 0;

    std::string activation;

    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;

    std::size_t hidden_get() const override;
};

struct conv_layer : layer {
    std::size_t c  = 0;
    std::size_t v1 = 0;
    std::size_t v2 = 0;
    std::size_t k  = 0;
    std::size_t w1 = 0;
    std::size_t w2 = 0;

    std::string activation;

    void print(std::ostream& out) const override;
    bool parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) override;

    bool is_conv() const override;
    std::size_t hidden_get() const override;
    std::size_t hidden_get_1() const override;
    std::size_t hidden_get_2() const override;
    std::size_t hidden_get_3() const override;
};

} //end of namespace dllp
