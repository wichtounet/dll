//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "layer.hpp"
#include "parse_utils.hpp"

void dllp::rbm_layer::print(std::ostream& out) const {
    out << "dll::rbm_desc<" << visible << ", " << hidden;

    if(!visible_unit.empty()){
        out << ", dll::visible<dll::unit_type::" << unit_type(visible_unit) << ">";
    }

    if(!hidden_unit.empty()){
        out << ", dll::hidden<dll::unit_type::" << unit_type(hidden_unit) << ">";
    }

    if(!decay.empty()){
        out << ", dll::weight_decay<dll::decay_type::" << decay_to_str(decay) << ">\n";
    }

    if(batch_size > 0){
        out << ", dll::batch_size<" << batch_size << ">";
    }

    if(momentum != dll::processor::stupid_default){
        out << ", dll::momentum";
    }

    if(parallel_mode){
        out << ", dll::parallel_mode";
    }

    out << ">::rbm_t";
}

bool dllp::rbm_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) {
    while(i < lines.size()){
        if(dllp::starts_with(lines[i], "visible:")){
            visible = std::stol(dllp::extract_value(lines[i], "visible: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "hidden:")){
            hidden = std::stol(dllp::extract_value(lines[i], "hidden: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "batch:")){
            batch_size = std::stol(dllp::extract_value(lines[i], "batch: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "momentum:")){
            momentum = std::stod(dllp::extract_value(lines[i], "momentum: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "parallel_mode:")){
            parallel_mode = dllp::extract_value(lines[i], "parallel_mode: ") == "true";
            ++i;
        } else if(dllp::starts_with(lines[i], "learning_rate:")){
            learning_rate = std::stod(dllp::extract_value(lines[i], "learning_rate: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "weight_decay:")){
            decay = dllp::extract_value(lines[i], "weight_decay: ");
            ++i;
        } else if(dllp::starts_with(lines[i], "l1_weight_cost:")){
            l1_weight_cost = std::stod(dllp::extract_value(lines[i], "l1_weight_cost: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "l2_weight_cost:")){
            l2_weight_cost = std::stod(dllp::extract_value(lines[i], "l2_weight_cost: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "hidden_unit:")){
            hidden_unit = dllp::extract_value(lines[i], "hidden_unit: ");
            ++i;

            if(!dllp::valid_unit(hidden_unit)){
                std::cout << "dllp: error: invalid hidden unit type must be one of [binary, softmax, gaussian]" << std::endl;
                return false;
            }
        } else if(dllp::starts_with(lines[i], "visible_unit:")){
            visible_unit = dllp::extract_value(lines[i], "visible_unit: ");
            ++i;

            if(!dllp::valid_unit(visible_unit)){
                std::cout << "dllp: error: invalid visible unit type must be one of [binary, softmax, gaussian]" << std::endl;
                return false;
            }
        } else {
            break;
        }
    }

    if(layers.empty() && !visible){
        std::cout << "dllp: error: The first layer needs number of visible units" << std::endl;
        return false;
    }

    if(!hidden){
        std::cout << "dllp: error: The number of hidden units is mandatory" << std::endl;
        return false;
    }

    if(!layers.empty()){
        visible = layers.back()->hidden_get();
    }

    return true;
}

void dllp::rbm_layer::set(std::ostream& out, const std::string& lhs) const {
    if(learning_rate != dll::processor::stupid_default){
        out << lhs << ".learning_rate = " << learning_rate << ";\n";
    }

    if(momentum != dll::processor::stupid_default){
        out << lhs << ".initial_momentum = " << momentum << ";\n";
        out << lhs << ".final_momentum = " << momentum << ";\n";
    }

    if(l1_weight_cost != dll::processor::stupid_default){
        out << lhs << ".l1_weight_cost = " << l1_weight_cost << ";\n";
    }

    if(l2_weight_cost != dll::processor::stupid_default){
        out << lhs << ".l2_weight_cost = " << l2_weight_cost << ";\n";
    }
}

std::size_t dllp::rbm_layer::hidden_get() const {
    return hidden;
}

bool dllp::conv_rbm_layer::is_conv() const {
    return true;
}

void dllp::conv_rbm_layer::print(std::ostream& out) const {
    out << "dll::conv_rbm_desc<" << c << ", " << v1 << ", " << v2 << ", " << k << ", " << (v1 - w1 + 1) << ", " << (v2 - w2 + 1);

    if(!visible_unit.empty()){
        out << ", dll::visible<dll::unit_type::" << unit_type(visible_unit) << ">";
    }

    if(!hidden_unit.empty()){
        out << ", dll::hidden<dll::unit_type::" << unit_type(hidden_unit) << ">";
    }

    if(batch_size > 0){
        out << ", dll::batch_size<" << batch_size << ">";
    }

    if(momentum != dll::processor::stupid_default){
        out << ", dll::momentum";
    }

    if(parallel_mode){
        out << ", dll::parallel_mode";
    }

    if(!decay.empty()){
        out << ", dll::weight_decay<dll::decay_type::" << decay_to_str(decay) << ">\n";
    }

    out << ">::rbm_t";
}

bool dllp::conv_rbm_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) {
    while(i < lines.size()){
        if(dllp::starts_with(lines[i], "channels:")){
            c = std::stol(dllp::extract_value(lines[i], "channels: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "filters:")){
            k = std::stol(dllp::extract_value(lines[i], "filters: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "v1:")){
            v1 = std::stol(dllp::extract_value(lines[i], "v1: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "v2:")){
            v2 = std::stol(dllp::extract_value(lines[i], "v2: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "w1:")){
            w1 = std::stol(dllp::extract_value(lines[i], "w1: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "w2:")){
            w2 = std::stol(dllp::extract_value(lines[i], "w2: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "batch:")){
            batch_size = std::stol(dllp::extract_value(lines[i], "batch: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "momentum:")){
            momentum = std::stod(dllp::extract_value(lines[i], "momentum: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "parallel_mode:")){
            parallel_mode = dllp::extract_value(lines[i], "parallel_mode: ") == "true";
            ++i;
        } else if(dllp::starts_with(lines[i], "learning_rate:")){
            learning_rate = std::stod(dllp::extract_value(lines[i], "learning_rate: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "weight_decay:")){
            decay = dllp::extract_value(lines[i], "weight_decay: ");
            ++i;
        } else if(dllp::starts_with(lines[i], "l1_weight_cost:")){
            l1_weight_cost = std::stod(dllp::extract_value(lines[i], "l1_weight_cost: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "l2_weight_cost:")){
            l2_weight_cost = std::stod(dllp::extract_value(lines[i], "l2_weight_cost: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "hidden_unit:")){
            hidden_unit = dllp::extract_value(lines[i], "hidden_unit: ");
            ++i;

            if(!dllp::valid_unit(hidden_unit)){
                std::cout << "dllp: error: invalid hidden unit type must be one of [binary, softmax, gaussian]" << std::endl;
                return false;
            }
        } else if(dllp::starts_with(lines[i], "visible_unit:")){
            visible_unit = dllp::extract_value(lines[i], "visible_unit: ");
            ++i;

            if(!dllp::valid_unit(visible_unit)){
                std::cout << "dllp: error: invalid visible unit type must be one of [binary, softmax, gaussian]" << std::endl;
                return false;
            }
        } else {
            break;
        }
    }

    if(layers.empty() && (!c || !v1 || !v2 || !k || !w1 || !w2)){
        std::cout << "dllp: error: The first layer needs input and output sizes" << std::endl;
        return false;
    } else if(!layers.empty() && !k){
        std::cout << "dllp: error: The number of filters is mandatory" << std::endl;
        return false;
    } else if(!layers.empty() && (!w1 || !w2)){
        std::cout << "dllp: error: The size of the filters is mandatory" << std::endl;
        return false;
    }

    if(!layers.empty()){
        c = layers.back()->hidden_get_1();
        v1 = layers.back()->hidden_get_2();
        v2 = layers.back()->hidden_get_3();
    }

    return true;
}

void dllp::conv_rbm_layer::set(std::ostream& out, const std::string& lhs) const {
    if(learning_rate != dll::processor::stupid_default){
        out << lhs << ".learning_rate = " << learning_rate << ";\n";
    }

    if(momentum != dll::processor::stupid_default){
        out << lhs << ".initial_momentum = " << momentum << ";\n";
        out << lhs << ".final_momentum = " << momentum << ";\n";
    }

    if(l1_weight_cost != dll::processor::stupid_default){
        out << lhs << ".l1_weight_cost = " << l1_weight_cost << ";\n";
    }

    if(l2_weight_cost != dll::processor::stupid_default){
        out << lhs << ".l2_weight_cost = " << l2_weight_cost << ";\n";
    }
}

std::size_t dllp::conv_rbm_layer::hidden_get() const {
    return k * (v1 - w1 + 1) * (v2 - w2 + 1);
}

std::size_t dllp::conv_rbm_layer::hidden_get_1() const {
    return k;
}

std::size_t dllp::conv_rbm_layer::hidden_get_2() const {
    return v1 - w1 + 1;
}

std::size_t dllp::conv_rbm_layer::hidden_get_3() const {
    return v2 - w2 + 1;
}

void dllp::dense_layer::print(std::ostream& out) const {
    out << "dll::dense_desc<" << visible << ", " << hidden;

    if(!activation.empty()){
        out << ", dll::activation<dll::function::" << activation_function(activation) << ">";
    }

    out << ">::layer_t";
}

bool dllp::dense_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) {
    while(i < lines.size()){
        if(dllp::starts_with(lines[i], "visible:")){
            visible = std::stol(dllp::extract_value(lines[i], "visible: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "hidden:")){
            hidden = std::stol(dllp::extract_value(lines[i], "hidden: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "activation:")){
            activation = dllp::extract_value(lines[i], "activation: ");
            ++i;

            if(!dllp::valid_activation(activation)){
                std::cout << "dllp: error: invalid activation function, must be [sigmoid,tanh,relu,softmax]" << std::endl;
                return false;
            }
        } else {
            break;
        }
    }

    if(layers.empty() && (visible == 0 || hidden == 0)){
        std::cout << "dllp: error: The first layer needs visible and hidden sizes" << std::endl;
        return false;
    } else if(!layers.empty() && hidden == 0){
        std::cout << "dllp: error: The number of hidden units is mandatory" << std::endl;
        return false;
    }

    if(!layers.empty()){
        visible = layers.back()->hidden_get();
    }

    return true;
}

std::size_t dllp::dense_layer::hidden_get() const {
    return hidden;
}

bool dllp::conv_layer::is_conv() const {
    return true;
}

void dllp::conv_layer::print(std::ostream& out) const {
    out << "dll::conv_desc<" << c << ", " << v1 << ", " << v2 << ", " << k << ", " << (v1 - w1 + 1) << ", " << (v2 - w2 + 1);

    if(!activation.empty()){
        out << ", dll::activation<dll::function::" << activation_function(activation) << ">";
    }

    out << ">::layer_t";
}

bool dllp::conv_layer::parse(const layers_t& layers, const std::vector<std::string>& lines, std::size_t& i) {
    while(i < lines.size()){
        if(dllp::starts_with(lines[i], "channels:")){
            c = std::stol(dllp::extract_value(lines[i], "channels: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "filters:")){
            k = std::stol(dllp::extract_value(lines[i], "filters: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "v1:")){
            v1 = std::stol(dllp::extract_value(lines[i], "v1: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "v2:")){
            v2 = std::stol(dllp::extract_value(lines[i], "v2: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "w1:")){
            w1 = std::stol(dllp::extract_value(lines[i], "w1: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "w2:")){
            w2 = std::stol(dllp::extract_value(lines[i], "w2: "));
            ++i;
        } else if(dllp::starts_with(lines[i], "activation:")){
            activation = dllp::extract_value(lines[i], "activation: ");
            ++i;

            if(!dllp::valid_activation(activation)){
                std::cout << "dllp: error: invalid activation function, must be [sigmoid,tanh,relu,softmax]" << std::endl;
                return false;
            }
        } else {
            break;
        }
    }

    if(layers.empty() && (!c || !v1 || !v2 || !k || !w1 || !w2)){
        std::cout << "dllp: error: The first layer needs input and output sizes" << std::endl;
        return false;
    } else if(!layers.empty() && !k){
        std::cout << "dllp: error: The number of filters is mandatory" << std::endl;
        return false;
    } else if(!layers.empty() && (!w1 || !w2)){
        std::cout << "dllp: error: The size of the filters is mandatory" << std::endl;
        return false;
    }

    if(!layers.empty()){
        c = layers.back()->hidden_get_1();
        v1 = layers.back()->hidden_get_2();
        v2 = layers.back()->hidden_get_3();
    }

    return true;
}

std::size_t dllp::conv_layer::hidden_get() const {
    return k * (v1 - w1 + 1) * (v2 - w2 + 1);
}

std::size_t dllp::conv_layer::hidden_get_1() const {
    return k;
}

std::size_t dllp::conv_layer::hidden_get_2() const {
    return v1 - w1 + 1;
}

std::size_t dllp::conv_layer::hidden_get_3() const {
    return v2 - w2 + 1;
}
