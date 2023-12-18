//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "layer.hpp"
#include "parse_utils.hpp"

dllp::parse_result dllp::base_rbm_layer::base_parse(const std::vector<std::string>& lines, size_t& i) {
    std::string value;

    if (dllp::extract_value(lines[i], "batch: ", value)) {
        batch_size = std::stol(value);
        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "momentum: ", value)) {
        momentum = std::stod(value);
        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "sparsity_target: ", value)) {
        sparsity_target = std::stod(value);
        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "shuffle: ", value)) {
        shuffle = value == "true";
        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "trainer: ", trainer)) {
        if (!dllp::valid_trainer(trainer)) {
            std::cout << "dllp: error: invalid trainer must be one of [cd, pcd]" << std::endl;
            return parse_result::ERROR;
        }

        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "sparsity: ", sparsity)) {
        if (!dllp::valid_sparsity(sparsity)) {
            std::cout << "dllp: error: invalid sparsity must be one of [local, global, lee]" << std::endl;
            return parse_result::ERROR;
        }

        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "learning_rate: ", value)) {
        learning_rate = std::stod(value);
        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "weight_decay: ", decay)) {
        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "l1_weight_cost: ", value)) {
        l1_weight_cost = std::stod(value);
        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "l2_weight_cost: ", value)) {
        l2_weight_cost = std::stod(value);
        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "pbias: ", value)) {
        pbias = std::stod(value);
        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "pbias_lambda: ", value)) {
        pbias_lambda = std::stod(value);
        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "hidden_unit: ", hidden_unit)) {
        if (!dllp::valid_unit(hidden_unit)) {
            std::cout << "dllp: error: invalid hidden unit type must be one of [binary, softmax, gaussian]" << std::endl;
            return parse_result::ERROR;
        }

        return parse_result::PARSED;
    } else if (dllp::extract_value(lines[i], "visible_unit: ", visible_unit)) {
        if (!dllp::valid_unit(visible_unit)) {
            std::cout << "dllp: error: invalid visible unit type must be one of [binary, softmax, gaussian]" << std::endl;
            return parse_result::ERROR;
        }

        return parse_result::PARSED;
    }

    return parse_result::NOT_PARSED;
}

void dllp::base_rbm_layer::set(std::ostream& out, const std::string& lhs) const {
    if (learning_rate != dll::processor::stupid_default) {
        out << lhs << ".learning_rate = " << learning_rate << ";\n";
    }

    if (momentum != dll::processor::stupid_default) {
        out << lhs << ".initial_momentum = " << momentum << ";\n";
        out << lhs << ".final_momentum = " << momentum << ";\n";
    }

    if (l1_weight_cost != dll::processor::stupid_default) {
        out << lhs << ".l1_weight_cost = " << l1_weight_cost << ";\n";
    }

    if (l2_weight_cost != dll::processor::stupid_default) {
        out << lhs << ".l2_weight_cost = " << l2_weight_cost << ";\n";
    }

    if (sparsity_target != dll::processor::stupid_default) {
        out << lhs << ".sparsity_target = " << sparsity_target << ";\n";
    }

    if (pbias != dll::processor::stupid_default) {
        out << lhs << ".pbias = " << pbias << ";\n";
    }

    if (pbias_lambda != dll::processor::stupid_default) {
        out << lhs << ".pbias_lambda = " << pbias_lambda << ";\n";
    }
}

void dllp::base_rbm_layer::print(std::ostream& out) const {
    out << "\n  , dll::weight_decay<dll::decay_type::" << decay_to_str(decay) << ">";
    out << "\n  , dll::sparsity<dll::sparsity_method::" << sparsity_to_str(sparsity) << ">";

    if (!visible_unit.empty()) {
        out << "\n  , dll::visible<dll::unit_type::" << unit_type(visible_unit) << ">";
    }

    if (!hidden_unit.empty()) {
        out << "\n  , dll::hidden<dll::unit_type::" << unit_type(hidden_unit) << ">";
    }

    if (batch_size > 0) {
        out << "\n  , dll::batch_size<" << batch_size << ">";
    }

    if (momentum != dll::processor::stupid_default) {
        out << "\n  , dll::momentum";
    }

    if (trainer == "pcd") {
        out << "\n  , dll::trainer_rbm<dll::pcd1_trainer_t>";
    }

    if (shuffle) {
        out << "\n  , dll::shuffle";
    }
}

void dllp::rbm_layer::print(std::ostream& out) const {
    out << "dll::rbm_desc<" << visible << ", " << hidden;

    base_rbm_layer::print(out);

    out << ">::layer_t";
}

bool dllp::rbm_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, size_t& i) {
    std::string value;

    while (i < lines.size()) {
        auto result = base_parse(lines, i);

        if (result == dllp::parse_result::PARSED) {
            ++i;
            continue;
        } else if (result == dllp::parse_result::ERROR) {
            return false;
        }

        if (dllp::extract_value(lines[i], "visible: ", value)) {
            visible = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "hidden: ", value)) {
            hidden = std::stol(value);
            ++i;
        } else {
            break;
        }
    }

    if (layers.empty() && !visible) {
        std::cout << "dllp: error: The first layer needs number of visible units" << std::endl;
        return false;
    }

    if (!hidden) {
        std::cout << "dllp: error: The number of hidden units is mandatory" << std::endl;
        return false;
    }

    if (!layers.empty()) {
        size_t i = layers.size() - 1;

        while(layers[i]->is_transform() && i > 0){
            --i;
        }

        visible = layers[i]->hidden_get();
    }

    return true;
}

size_t dllp::rbm_layer::hidden_get() const {
    return hidden;
}

bool dllp::conv_rbm_layer::is_conv() const {
    return true;
}

void dllp::conv_rbm_layer::print(std::ostream& out) const {
    out << "dll::conv_rbm_desc<" << c << ", " << v1 << ", " << v2 << ", " << k << ", " << w1 << ", " << w2;

    base_rbm_layer::print(out);

    out << ">::layer_t";
}

bool dllp::conv_rbm_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, size_t& i) {
    std::string value;

    while (i < lines.size()) {
        auto result = base_parse(lines, i);

        if (result == dllp::parse_result::PARSED) {
            ++i;
            continue;
        } else if (result == dllp::parse_result::ERROR) {
            return false;
        }

        if (dllp::extract_value(lines[i], "channels: ", value)) {
            c = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "filters: ", value)) {
            k = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "v1: ", value)) {
            v1 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "v2: ", value)) {
            v2 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "w1: ", value)) {
            w1 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "w2: ", value)) {
            w2 = std::stol(value);
            ++i;
        } else {
            break;
        }
    }

    if (layers.empty() && (!c || !v1 || !v2 || !k || !w1 || !w2)) {
        std::cout << "dllp: error: The first layer needs input and output sizes" << std::endl;
        return false;
    } else if (!layers.empty() && !k) {
        std::cout << "dllp: error: The number of filters is mandatory" << std::endl;
        return false;
    } else if (!layers.empty() && (!w1 || !w2)) {
        std::cout << "dllp: error: The size of the filters is mandatory" << std::endl;
        return false;
    }

    if (!layers.empty()) {
        size_t i = layers.size() - 1;

        while(layers[i]->is_transform() && i > 0){
            --i;
        }

        c  = layers[i]->hidden_get_1();
        v1 = layers[i]->hidden_get_2();
        v2 = layers[i]->hidden_get_3();
    }

    return true;
}

size_t dllp::conv_rbm_layer::hidden_get() const {
    return k * (v1 - w1 + 1) * (v2 - w2 + 1);
}

size_t dllp::conv_rbm_layer::hidden_get_1() const {
    return k;
}

size_t dllp::conv_rbm_layer::hidden_get_2() const {
    return v1 - w1 + 1;
}

size_t dllp::conv_rbm_layer::hidden_get_3() const {
    return v2 - w2 + 1;
}

bool dllp::conv_rbm_mp_layer::is_conv() const {
    return true;
}

void dllp::conv_rbm_mp_layer::print(std::ostream& out) const {
    out << "dll::conv_rbm_mp_desc<" << c << ", " << v1 << ", " << v2 << ", " << k << ", " << w1 << ", " << w2 << ", " << p;

    base_rbm_layer::print(out);

    out << ">::layer_t";
}

bool dllp::conv_rbm_mp_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, size_t& i) {
    std::string value;

    while (i < lines.size()) {
        auto result = base_parse(lines, i);

        if (result == dllp::parse_result::PARSED) {
            ++i;
            continue;
        } else if (result == dllp::parse_result::ERROR) {
            return false;
        }

        if (dllp::extract_value(lines[i], "channels: ", value)) {
            c = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "filters: ", value)) {
            k = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "pool: ", value)) {
            p = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "v1: ", value)) {
            v1 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "v2: ", value)) {
            v2 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "w1: ", value)) {
            w1 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "w2: ", value)) {
            w2 = std::stol(value);
            ++i;
        } else {
            break;
        }
    }

    if (layers.empty() && (!c || !v1 || !v2 || !k || !w1 || !w2)) {
        std::cout << "dllp: error: The first layer needs input and output sizes" << std::endl;
        return false;
    } else if (!p) {
        std::cout << "dllp: error: The pool parameter is mandatory" << std::endl;
        return false;
    } else if (!layers.empty() && !k) {
        std::cout << "dllp: error: The number of filters is mandatory" << std::endl;
        return false;
    } else if (!layers.empty() && (!w1 || !w2)) {
        std::cout << "dllp: error: The size of the filters is mandatory" << std::endl;
        return false;
    }

    if (!layers.empty()) {
        size_t i = layers.size() - 1;

        while(layers[i]->is_transform() && i > 0){
            --i;
        }

        c  = layers[i]->hidden_get_1();
        v1 = layers[i]->hidden_get_2();
        v2 = layers[i]->hidden_get_3();
    }

    return true;
}

size_t dllp::conv_rbm_mp_layer::hidden_get() const {
    return k * ((v1 - w1 + 1) / p) * ((v2 - w2 + 1) / p);
}

size_t dllp::conv_rbm_mp_layer::hidden_get_1() const {
    return k;
}

size_t dllp::conv_rbm_mp_layer::hidden_get_2() const {
    return (v1 - w1 + 1) / p;
}

size_t dllp::conv_rbm_mp_layer::hidden_get_3() const {
    return (v2 - w2 + 1) / p;
}

void dllp::dense_layer::print(std::ostream& out) const {
    out << "dll::dense_layer_desc<" << visible << ", " << hidden;

    if (!activation.empty()) {
        out << "\n  , dll::activation<dll::function::" << activation_function(activation) << ">";
    }

    out << ">::layer_t";
}

bool dllp::dense_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, size_t& i) {
    std::string value;

    while (i < lines.size()) {
        if (dllp::extract_value(lines[i], "visible: ", value)) {
            visible = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "hidden: ", value)) {
            hidden = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "activation: ", activation)) {
            ++i;

            if (!dllp::valid_activation(activation)) {
                std::cout << "dllp: error: invalid activation function, must be [sigmoid,tanh,relu,softmax]" << std::endl;
                return false;
            }
        } else {
            break;
        }
    }

    if (layers.empty() && (visible == 0 || hidden == 0)) {
        std::cout << "dllp: error: The first layer needs visible and hidden sizes" << std::endl;
        return false;
    } else if (!layers.empty() && hidden == 0) {
        std::cout << "dllp: error: The number of hidden units is mandatory" << std::endl;
        return false;
    }

    if (!layers.empty()) {
        size_t i = layers.size() - 1;

        while(layers[i]->is_transform() && i > 0){
            --i;
        }

        visible = layers[i]->hidden_get();
    }

    return true;
}

size_t dllp::dense_layer::hidden_get() const {
    return hidden;
}

bool dllp::conv_layer::is_conv() const {
    return true;
}

void dllp::conv_layer::print(std::ostream& out) const {
    out << "dll::conv_layer_desc<" << c << ", " << v1 << ", " << v2 << ", " << k << ", " << w1 << ", " << w2;

    if (!activation.empty()) {
        out << "\n  , dll::activation<dll::function::" << activation_function(activation) << ">";
    }

    out << ">::layer_t";
}

bool dllp::conv_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, size_t& i) {
    std::string value;

    while (i < lines.size()) {
        if (dllp::extract_value(lines[i], "channels: ", value)) {
            c = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "filters: ", value)) {
            k = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "v1: ", value)) {
            v1 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "v2: ", value)) {
            v2 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "w1: ", value)) {
            w1 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "w2: ", value)) {
            w2 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "activation: ", activation)) {
            ++i;

            if (!dllp::valid_activation(activation)) {
                std::cout << "dllp: error: invalid activation function, must be [sigmoid,tanh,relu,softmax]" << std::endl;
                return false;
            }
        } else {
            break;
        }
    }

    if (layers.empty() && (!c || !v1 || !v2 || !k || !w1 || !w2)) {
        std::cout << "dllp: error: The first layer needs input and output sizes" << std::endl;
        return false;
    } else if (!layers.empty() && !k) {
        std::cout << "dllp: error: The number of filters is mandatory" << std::endl;
        return false;
    } else if (!layers.empty() && (!w1 || !w2)) {
        std::cout << "dllp: error: The size of the filters is mandatory" << std::endl;
        return false;
    }

    if (!layers.empty()) {
        size_t i = layers.size() - 1;

        while(layers[i]->is_transform() && i > 0){
            --i;
        }

        c  = layers[i]->hidden_get_1();
        v1 = layers[i]->hidden_get_2();
        v2 = layers[i]->hidden_get_3();
    }

    return true;
}

size_t dllp::conv_layer::hidden_get() const {
    return k * (v1 - w1 + 1) * (v2 - w2 + 1);
}

size_t dllp::conv_layer::hidden_get_1() const {
    return k;
}

size_t dllp::conv_layer::hidden_get_2() const {
    return v1 - w1 + 1;
}

size_t dllp::conv_layer::hidden_get_3() const {
    return v2 - w2 + 1;
}

bool dllp::pooling_layer::is_conv() const {
    return true;
}

void dllp::pooling_layer::print(std::ostream& out) const {
    out << "<" << c << ", " << v1 << ", " << v2 << ", " << c1 << ", " << c2 << ", " << c3;
    out << ">::layer_t";
}

bool dllp::pooling_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, size_t& i) {
    std::string value;

    while (i < lines.size()) {
        if (dllp::extract_value(lines[i], "channels: ", value)) {
            c = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "v1: ", value)) {
            v1 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "v2: ", value)) {
            v2 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "c1: ", value)) {
            c1 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "c2: ", value)) {
            c2 = std::stol(value);
            ++i;
        } else if (dllp::extract_value(lines[i], "c3: ", value)) {
            c3 = std::stol(value);
            ++i;
        } else {
            break;
        }
    }

    if (layers.empty() && (!c || !v1 || !v2 || !c1 || !c2 || !c3)) {
        std::cout << "dllp: error: The first layer needs input and output sizes" << std::endl;
        return false;
    } else if (!layers.empty() && (!c1 || !c2 || !c3)) {
        std::cout << "dllp: error: The factors of the pooling is mandatory" << std::endl;
        return false;
    }

    if (!layers.empty()) {
        size_t i = layers.size() - 1;

        while(layers[i]->is_transform() && i > 0){
            --i;
        }

        c  = layers[i]->hidden_get_1();
        v1 = layers[i]->hidden_get_2();
        v2 = layers[i]->hidden_get_3();
    }

    return true;
}

size_t dllp::pooling_layer::hidden_get() const {
    return hidden_get_1() * hidden_get_2() * hidden_get_3();
}

size_t dllp::pooling_layer::hidden_get_1() const {
    return c / c1;
}

size_t dllp::pooling_layer::hidden_get_2() const {
    return v1 / c2;
}

size_t dllp::pooling_layer::hidden_get_3() const {
    return v2 / c3;
}

void dllp::mp_layer::print(std::ostream& out) const {
    out << "dll::mp_3d_layer_desc";
    pooling_layer::print(out);
}

bool dllp::mp_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, size_t& i) {
    return pooling_layer::parse(layers, lines, i);
}

void dllp::avgp_layer::print(std::ostream& out) const {
    out << "dll::avgp_3d_layer_desc";
    pooling_layer::print(out);
}

bool dllp::avgp_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, size_t& i) {
    return pooling_layer::parse(layers, lines, i);
}

bool dllp::function_layer::is_transform() const {
    return true;
}

void dllp::function_layer::print(std::ostream& out) const {
    out << "dll::activation_layer_desc<"
        << "dll::function::" << activation_function(activation)
        << ">::layer_t";
}

bool dllp::function_layer::parse(const layers_t& /*layers*/, const std::vector<std::string>& lines, size_t& i) {
    std::string value;

    while (i < lines.size()) {
        if (dllp::extract_value(lines[i], "activation: ", activation)) {
            ++i;

            if (!dllp::valid_activation(activation)) {
                std::cout << "dllp: error: invalid activation function, must be [sigmoid,tanh,relu,softmax]" << std::endl;
                return false;
            }
        } else {
            break;
        }
    }

    return true;
}
