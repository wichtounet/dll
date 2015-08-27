//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include<memory>

#include "cpp_utils/string.hpp"

namespace dllp {

struct layer {
    virtual void print(std::ostream& out) = 0;
};

struct rbm_layer : layer {
    std::size_t visible = 0;
    std::size_t hidden = 0;

    void print(std::ostream& out) override {
        out << "dll::rbm_layer<>::rbm_t \n";
    }
};

struct datasource {
    std::string source_file;
    std::string reader;
};

struct task {
    datasource pretraining;
    datasource samples;
    datasource labels;

    std::vector<std::shared_ptr<layer>> layers;
};

void print_usage(){
    std::cout << "Usage: dllp conf_file action" << std::endl;
}

bool starts_with(const std::string& str, const std::string& search){
    return std::mismatch(search.begin(), search.end(), str.begin()).first == search.end();
}

std::string extract_value(const std::string& str, const std::string& search){
    return {str.begin() + str.find(search) + search.size(), str.end()};
}

datasource parse_datasource(const std::vector<std::string>& lines, std::size_t& i){
    datasource source;

    source.reader = "default";

    while(i < lines.size()){
        if(starts_with(lines[i], "source: ")){
            source.source_file = extract_value(lines[i], "source: ");
            ++i;
        } else if(starts_with(lines[i], "reader: ")){
            source.reader = extract_value(lines[i], "reader: ");
            ++i;
        } else {
            break;
        }
    }

    if(source.source_file.empty()){
        std::cout << "dllp:: error: missing source" << std::endl;
    }

    return source;
}

void generate(task& t);

} //end of dllp namespace

int main(int argc, char* argv[]){
    if(argc < 3){
        std::cout << "dllp: Not enough arguments" << std::endl;
        dllp::print_usage();
        return 1;
    }

    std::string source_file(argv[1]);
    std::string action(argv[2]);

    std::ifstream source_stream(source_file);

    std::string current_line;
    std::vector<std::string> lines;

    while(std::getline(source_stream, current_line)) {
        lines.push_back(cpp::trim(current_line));
    }

    dllp::task t;

    for(std::size_t i = 0; i < lines.size();){
        auto& current_line = lines[i];

        if(current_line == "input:"){
            ++i;

            if(i == lines.size()){
                std::cout << "dllp: error: input expect at least one child" << std::endl;

                return 1;
            }

            while(i < lines.size()){
                if(lines[i] == "pretraining:"){
                    t.pretraining = dllp::parse_datasource(lines, ++i);
                } else if(lines[i] == "samples:"){
                    t.samples = dllp::parse_datasource(lines, ++i);
                } else if(lines[i] == "labels:"){
                    t.labels = dllp::parse_datasource(lines, ++i);
                } else {
                    break;
                }
            }
        } else if(current_line == "rbm:"){
            ++i;

            if(i == lines.size()){
                std::cout << "dllp: error: rbm expect at least visible and hidden parameters" << std::endl;

                return 1;
            }

            auto rbm = std::make_shared<dllp::rbm_layer>();

            while(i < lines.size()){
                if(dllp::starts_with(lines[i], "visible:")){
                    rbm->visible = std::stol(dllp::extract_value(lines[i], "visible: "));
                    ++i;
                } else if(dllp::starts_with(lines[i], "hidden:")){
                    rbm->hidden = std::stol(dllp::extract_value(lines[i], "hidden: "));
                    ++i;
                } else {
                    break;
                }
            }

            t.layers.push_back(std::move(rbm));
        } else {
            std::cout << "dllp: error: Invalid line: " << current_line << std::endl;

            return 1;
        }
    }

    dllp::generate(t);

    return 0;
}

namespace dllp {

void generate(task& t){
    std::ofstream out_stream(".dbn.cpp");

    //TODO

    for(auto& layer : t.layers){
        layer->print(out_stream);
    }
}

} //end of namespace dllp

