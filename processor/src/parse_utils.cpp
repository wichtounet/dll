//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "parse_utils.hpp"

bool dllp::starts_with(const std::string& str, const std::string& search){
    return std::mismatch(search.begin(), search.end(), str.begin()).first == search.end();
}

std::string dllp::extract_value(const std::string& str, const std::string& search){
    return {str.begin() + str.find(search) + search.size(), str.end()};
}

bool dllp::extract_value(const std::string& line, const std::string& search, std::string& value){
    if(starts_with(line, search)){
        value = extract_value(line, search);
        return true;
    }

    return false;
}

std::string dllp::unit_type(const std::string& unit){
    if(unit == "binary"){
        return "BINARY";
    } else if(unit == "softmax"){
        return "SOFTMAX";
    } else if(unit == "gaussian"){
        return "GAUSSIAN";
    } else {
        return "INVALID";
    }
}

std::string dllp::activation_function(const std::string& unit){
    if(unit == "sigmoid"){
        return "SIGMOID";
    } else if(unit == "tanh"){
        return "TANH";
    } else if(unit == "softmax"){
        return "SOFTMAX";
    } else if(unit == "relu"){
        return "RELU";
    } else {
        return "INVALID";
    }
}

std::string dllp::decay_to_str(const std::string& decay){
    if(decay == "l1"){
        return "L1";
    } else if(decay == "l1_full"){
        return "L1_FULL";
    } else if(decay == "l2"){
        return "L2";
    } else if(decay == "l2_full"){
        return "L2_FULL";
    } else if(decay == "l1l2"){
        return "L1L2";
    } else if(decay == "l1l2_full"){
        return "L1L2_FULL";
    } else if(decay == "none"){
        return "NONE";
    } else {
        return "INVALID";
    }
}

bool dllp::valid_unit(const std::string& unit){
    return unit == "binary" || unit == "softmax" || unit == "gaussian";
}

bool dllp::valid_trainer(const std::string& unit){
    return unit == "cd" || unit == "pcd";
}

bool dllp::valid_activation(const std::string& unit){
    return unit == "sigmoid" || unit == "softmax" || unit == "tanh" || unit == "relu";
}

std::vector<std::string> dllp::read_lines(const std::string& source_file){
    std::vector<std::string> lines;

    std::ifstream source_stream(source_file);

    std::string current_line;

    while(std::getline(source_stream, current_line)) {
        std::string processed(cpp::trim(current_line));

        if(!processed.empty()){
            lines.emplace_back(std::move(processed));
        }
    }

    return lines;
}
