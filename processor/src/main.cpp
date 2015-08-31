//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdlib>

#include "cpp_utils/string.hpp"

#include "dll/processor/processor.hpp"

namespace dllp {

struct layer {
    virtual void print(std::ostream& out) = 0;
};

struct rbm_layer : layer {
    std::size_t visible = 0;
    std::size_t hidden = 0;

    void print(std::ostream& out) override {
        out << "dll::rbm_desc<" << visible << ", " << hidden << ">::rbm_t";
    }
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

dll::processor::datasource parse_datasource(const std::vector<std::string>& lines, std::size_t& i){
    dll::processor::datasource source;

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

void generate(const std::vector<std::shared_ptr<dllp::layer>>& layers, dll::processor::task& t, const std::vector<std::string>& actions);
void compile(const char* cxx);

} //end of dllp namespace

int main(int argc, char* argv[]){
    if(argc < 3){
        std::cout << "dllp: Not enough arguments" << std::endl;
        dllp::print_usage();
        return 1;
    }

    const auto* cxx = std::getenv("CXX");

    if(!cxx){
        std::cout << "CXX environment variable must be set" << std::endl;
        return 2;
    }

    std::string source_file(argv[1]);

    std::vector<std::string> actions;

    for(int i = 2; i < argc; ++i){
        actions.emplace_back(argv[i]);
    }

    std::ifstream source_stream(source_file);

    std::string current_line;
    std::vector<std::string> lines;

    while(std::getline(source_stream, current_line)) {
        lines.push_back(cpp::trim(current_line));
    }

    dll::processor::task t;

    std::vector<std::shared_ptr<dllp::layer>> layers;

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

            layers.push_back(std::move(rbm));
        } else if(current_line == "pretraining:"){
            ++i;

            while(i < lines.size()){
                if(dllp::starts_with(lines[i], "epochs:")){
                    t.pt_desc.epochs = std::stol(dllp::extract_value(lines[i], "epochs: "));
                    ++i;
                } else {
                    break;
                }
            }
        } else if(current_line.empty()) {
            ++i;
        } else {
            std::cout << "dllp: error: invalid line: " << current_line << std::endl;

            return 1;
        }
    }

    //Generate the CPP file
    dllp::generate(layers, t, actions);

    //Compile the generate file
    dllp::compile(cxx);

    //Run the generated program

    std::cout << "Executing the program" << std::endl;

    auto exec_result = system("./dbn.out");

    if(exex_result){
        std::cout << "Impossible to execute the generated file" << std::endl;
        return exec_result;
    }

    return 0;
}

namespace dllp {

struct datasource {
    std::string source_file;
    std::string reader;

    bool binarize = false;
    bool normalize = false;
};

std::string datasource_to_string(const std::string& lhs, const dll::processor::datasource& ds){
    std::string result;

    result += lhs + ".source_file = \"" + ds.source_file + "\";";
    result += lhs + ".reader = \"" + ds.reader + "\";";
    result += lhs + ".binarize = \"" + (ds.binarize ? "true" : "false") + "\";";
    result += lhs + ".normalize = \"" + (ds.normalize ? "true" : "false") + "\";";

    return result;
}

std::string task_to_string(const std::string& name, const dll::processor::task& t){
    std::string result;

    result += "   dll::processor::task ";
    result += name;
    result += ";";
    result += datasource_to_string(name + ".pretraining", t.pretraining);
    result += "\n";
    result += datasource_to_string(name + ".samples", t.samples);
    result += "\n";
    result += datasource_to_string(name + ".labels", t.labels);
    result += "\n";

    //TODO Generalize that
    result += name + ".pt_desc.epochs = " + std::to_string(t.pt_desc.epochs) + ";";

    return result;
}

std::string vector_to_string(const std::string& name, const std::vector<std::string>& vec){
    std::string result;

    result += "   std::vector<std::string> ";
    result += name;
    result += "{";
    std::string comma = "";
    for(auto& value : vec){
        result += comma;
        result += "\"";
        result += value;
        result += "\"";
        comma = ", ";
    }
    result += "};";

    return result;
}

void generate(const std::vector<std::shared_ptr<dllp::layer>>& layers, dll::processor::task& t, const std::vector<std::string>& actions){
    std::ofstream out_stream(".dbn.cpp");

    out_stream << "#include <memory>\n";
    out_stream << "#include \"dll/processor/processor.hpp\"\n\n";

    out_stream << "using dbn_t = dll::dbn_desc<dll::dbn_layers<\n";

    //TODO

    std::string comma = "  ";

    for(auto& layer : layers){
        out_stream << comma;
        layer->print(out_stream);
        comma = "\n, ";
    }

    out_stream << "\n>>::dbn_t;\n\n";

    out_stream << "int main(int argc, char* argv[]){\n";
    out_stream << "   auto dbn = std::make_unique<dbn_t>();\n";
    out_stream << task_to_string("t", t) << "\n";
    out_stream << vector_to_string("actions", actions) << "\n";
    out_stream << "   dll::processor::execute(*dbn, t, actions);\n";
    out_stream << "}\n";
}

void compile(const char* cxx){
    std::cout << "Compiling the program..." << std::endl;

    std::string compile_command(cxx);

    compile_command += " -o .dbn.out ";
    compile_command += " -std=c++1y ";
    compile_command += " -pthread ";
    compile_command += " .dbn.cpp ";

    int compile_result = system(compile_command.c_str());

    if(compile_result){
        std::cout << "Compilation failed" << std::endl;
    } else {
        std::cout << "... done" << std::endl;
    }
}

} //end of namespace dllp

