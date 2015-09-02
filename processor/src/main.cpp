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

#include <sys/stat.h>
#include <sys/types.h>

#include "cpp_utils/string.hpp"

#include "dll/processor/processor.hpp"

namespace dllp {

struct options {
    bool mkl = false;
    bool cache = false;
};

struct layer {
    virtual void print(std::ostream& out) const = 0;
    virtual std::size_t hidden_get() const = 0;

    virtual void set(std::ostream& /*out*/, const std::string& /*lhs*/) const { /* Nothing */ };
};

std::string unit_type(const std::string& unit){
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

bool valid_unit(const std::string& unit){
    return unit == "binary" || unit == "softmax" || unit == "gaussian";
}

struct rbm_layer : layer {
    std::size_t visible = 0;
    std::size_t hidden = 0;

    std::string visible_unit;
    std::string hidden_unit;

    double learning_rate = dll::processor::stupid_default;
    double momentum = dll::processor::stupid_default;
    std::size_t batch_size = 0;

    void print(std::ostream& out) const override {
        out << "dll::rbm_desc<" << visible << ", " << hidden;

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

        out << ">::rbm_t";
    }

    virtual void set(std::ostream& out, const std::string& lhs) const override {
        if(learning_rate != dll::processor::stupid_default){
            out << lhs << ".learning_rate = " << learning_rate << ";\n";
        }

        if(momentum != dll::processor::stupid_default){
            out << lhs << ".initial_momentum = " << momentum << ";\n";
            out << lhs << ".final_momentum = " << momentum << ";\n";
        }
    }

    std::size_t hidden_get() const override {
        return hidden;
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
        } else if(starts_with(lines[i], "binarize: ")){
            source.binarize = extract_value(lines[i], "binarize: ") == "true" ? true : false;
            ++i;
        } else if(starts_with(lines[i], "normalize: ")){
            source.normalize = extract_value(lines[i], "normalize: ") == "true" ? true : false;
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

void parse_datasource_pack(dll::processor::datasource_pack& pack, const std::vector<std::string>& lines, std::size_t& i){
    std::size_t limit = -1;
    while(i < lines.size()){
        if(starts_with(lines[i], "samples:")){
            pack.samples = parse_datasource(lines, ++i);
        } else if(starts_with(lines[i], "labels:")){
            pack.labels = parse_datasource(lines, ++i);
        } else if(starts_with(lines[i], "limit:")){
            limit = std::stol(extract_value(lines[i], "limit: "));
            ++i;
        } else {
            break;
        }
    }

    pack.samples.limit = limit;
    pack.labels.limit = limit;
}

void generate(const std::vector<std::shared_ptr<dllp::layer>>& layers, dll::processor::task& t, const std::vector<std::string>& actions);
bool compile(const char* cxx, const options& opt);

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

    dllp::options opt;

    std::size_t i = 1;

    while(true){
        if(std::string(argv[i]) == "--mkl"){
            opt.mkl = true;
            ++i;
        } else if(std::string(argv[i]) == "--cache"){
            opt.cache = true;
            ++i;
        } else {
            break;
        }
    }

    std::string source_file(argv[i++]);

    std::vector<std::string> actions;

    for(; i < std::size_t(argc); ++i){
        actions.emplace_back(argv[i]);
    }

    std::ifstream source_stream(source_file);

    std::string current_line;
    std::vector<std::string> lines;

    while(std::getline(source_stream, current_line)) {
        std::string processed(cpp::trim(current_line));

        if(!processed.empty()){
            lines.emplace_back(std::move(processed));
        }
    }

    dll::processor::task t;

    std::vector<std::shared_ptr<dllp::layer>> layers;

    for(std::size_t i = 0; i < lines.size();){
        auto& current_line = lines[i];

        if(current_line == "data:"){
            ++i;

            while(i < lines.size()){
                if(lines[i] == "pretraining:"){
                    dllp::parse_datasource_pack(t.pretraining, lines, ++i);
                } else if(lines[i] == "training:"){
                    dllp::parse_datasource_pack(t.training, lines, ++i);
                } else if(lines[i] == "testing:"){
                    dllp::parse_datasource_pack(t.testing, lines, ++i);
                } else {
                    break;
                }
            }
        } else if(current_line == "network:"){
            ++i;

            while(i < lines.size()){
                if(lines[i] == "rbm:"){
                    ++i;

                    auto rbm = std::make_shared<dllp::rbm_layer>();

                    while(i < lines.size()){
                        if(dllp::starts_with(lines[i], "visible:")){
                            rbm->visible = std::stol(dllp::extract_value(lines[i], "visible: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "hidden:")){
                            rbm->hidden = std::stol(dllp::extract_value(lines[i], "hidden: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "batch:")){
                            rbm->batch_size = std::stol(dllp::extract_value(lines[i], "batch: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "momentum:")){
                            rbm->momentum = std::stod(dllp::extract_value(lines[i], "momentum: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "learning_rate:")){
                            rbm->learning_rate = std::stod(dllp::extract_value(lines[i], "learning_rate: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "hidden_unit:")){
                            rbm->hidden_unit = dllp::extract_value(lines[i], "hidden_unit: ");
                            ++i;

                            if(!dllp::valid_unit(rbm->hidden_unit)){
                                std::cout << "dllp: error: invalid hidden unit type must be one of [binary, softmax, gaussian]" << std::endl;
                                return 1;
                            }
                        } else if(dllp::starts_with(lines[i], "visible_unit:")){
                            rbm->visible_unit = dllp::extract_value(lines[i], "visible_unit: ");
                            ++i;

                            if(!dllp::valid_unit(rbm->visible_unit)){
                                std::cout << "dllp: error: invalid visible unit type must be one of [binary, softmax, gaussian]" << std::endl;
                                return 1;
                            }
                        } else {
                            break;
                        }
                    }

                    if(layers.empty() && (rbm->visible == 0 || rbm->hidden == 0)){
                        std::cout << "dllp: error: The first layer needs visible and hidden sizes" << std::endl;
                    } else if(!layers.empty() && rbm->hidden == 0){
                        std::cout << "dllp: error: The number of hidden units is mandatory" << std::endl;
                    }

                    if(!layers.empty()){
                        rbm->visible = layers.back()->hidden_get();
                    }

                    layers.push_back(std::move(rbm));
                } else {
                    break;
                }
            }

        } else if(current_line == "options:"){
            ++i;

            while(i < lines.size()){
                if(lines[i] == "pretraining:"){
                    ++i;

                    while(i < lines.size()){
                        if(dllp::starts_with(lines[i], "epochs:")){
                            t.pt_desc.epochs = std::stol(dllp::extract_value(lines[i], "epochs: "));
                            ++i;
                        } else {
                            break;
                        }
                    }
                } else if(lines[i] == "training:"){
                    ++i;

                    while(i < lines.size()){
                        if(dllp::starts_with(lines[i], "epochs:")){
                            t.ft_desc.epochs = std::stol(dllp::extract_value(lines[i], "epochs: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "learning_rate:")){
                            t.ft_desc.learning_rate = std::stod(dllp::extract_value(lines[i], "learning_rate: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "momentum:")){
                            t.ft_desc.momentum = std::stod(dllp::extract_value(lines[i], "momentum: "));
                            ++i;
                        } else {
                            break;
                        }
                    }
                } else if(lines[i] == "weights:"){
                    ++i;

                    while(i < lines.size()){
                        if(dllp::starts_with(lines[i], "file:")){
                            t.w_desc.file = dllp::extract_value(lines[i], "file: ");
                            ++i;
                        } else {
                            break;
                        }
                    }
                } else {
                    break;
                }
            }
        } else {
            std::cout << "dllp: error: invalid line: " << i << ":" << current_line << std::endl;

            return 1;
        }
    }

    bool process = true;

    if(opt.cache){
        struct stat attr_conf;
        struct stat attr_exec;

        if(!stat(source_file.c_str(), &attr_conf)){
            if(!stat("./.dbn.out", &attr_exec)){
                auto mtime_conf = attr_conf.st_mtime;
                auto mtime_exec = attr_exec.st_mtime;

                if(mtime_exec > mtime_conf){
                    std::cout << "Skip compilation" << std::endl;
                    process = false;
                }
            }
        }
    }

    if(process){
        //Generate the CPP file
        dllp::generate(layers, t, actions);

        //Compile the generate file
        if(!dllp::compile(cxx, opt)){
            return 1;
        }
    }

    //Run the generated program

    std::cout << "Executing the program" << std::endl;

    auto exec_result = system("./.dbn.out");

    if(exec_result){
        std::cout << "Impossible to execute the generated file" << std::endl;
        return exec_result;
    }

    return 0;
}

namespace dllp {

std::string datasource_to_string(const std::string& lhs, const dll::processor::datasource& ds){
    std::string result;

    result += lhs + ".source_file = \"" + ds.source_file + "\";\n";
    result += lhs + ".reader = \"" + ds.reader + "\";\n";
    result += lhs + ".binarize = " + (ds.binarize ? "true" : "false") + ";\n";
    result += lhs + ".normalize = " + (ds.normalize ? "true" : "false") + ";\n";
    result += lhs + ".limit = " + std::to_string(ds.limit) + ";\n";

    return result;
}

std::string pt_desc_to_string(const std::string& lhs, const dll::processor::pretraining_desc& desc){
    std::string result;

    result += lhs + ".epochs = " + std::to_string(desc.epochs) + ";";

    return result;
}

std::string ft_desc_to_string(const std::string& lhs, const dll::processor::training_desc& desc){
    std::string result;

    result += lhs + ".epochs = " + std::to_string(desc.epochs) + ";";

    return result;
}

std::string w_desc_to_string(const std::string& lhs, const dll::processor::weights_desc& desc){
    std::string result;

    result += lhs + ".file = \"" + desc.file + "\";";

    return result;
}

std::string task_to_string(const std::string& name, const dll::processor::task& t){
    std::string result;

    result += "   dll::processor::task ";
    result += name;
    result += ";\n\n";
    result += datasource_to_string("   " + name + ".pretraining.samples", t.pretraining.samples);
    result += "\n";
    result += datasource_to_string("   " + name + ".training.samples", t.training.samples);
    result += "\n";
    result += datasource_to_string("   " + name + ".training.labels", t.training.labels);
    result += "\n";
    result += datasource_to_string("   " + name + ".testing.samples", t.testing.samples);
    result += "\n";
    result += datasource_to_string("   " + name + ".testing.labels", t.testing.labels);
    result += "\n";
    result += pt_desc_to_string("   " + name + ".pt_desc", t.pt_desc);
    result += "\n";
    result += ft_desc_to_string("   " + name + ".ft_desc", t.ft_desc);
    result += "\n";
    result += w_desc_to_string("   " + name + ".w_desc", t.w_desc);
    result += "\n";

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
    out_stream << "#include \"dll/processor/processor.hpp\"\n";
    out_stream << "#include \"dll/dense_stochastic_gradient_descent.hpp\"\n\n";

    out_stream << "using dbn_t = dll::dbn_desc<dll::dbn_layers<\n";

    std::string comma = "  ";

    for(auto& layer : layers){
        out_stream << comma;
        layer->print(out_stream);
        comma = "\n, ";
    }

    out_stream << "\n>";

    out_stream << ", dll::trainer<dll::dense_sgd_trainer>\n";

    if(t.ft_desc.momentum != dll::processor::stupid_default){
        out_stream << ", dll::momentum\n";
    }

    out_stream << ">::dbn_t;\n\n";

    out_stream << "int main(int argc, char* argv[]){\n";
    out_stream << "   auto dbn = std::make_unique<dbn_t>();\n";

    if(t.ft_desc.learning_rate != dll::processor::stupid_default){
        out_stream << "   dbn->learning_rate = " << t.ft_desc.learning_rate << ";\n";
    }

    if(t.ft_desc.momentum != dll::processor::stupid_default){
        out_stream << "   dbn->initial_momentum = " << t.ft_desc.momentum << ";\n";
        out_stream << "   dbn->final_momentum = " << t.ft_desc.momentum << ";\n";
    }

    for(std::size_t i = 0; i < layers.size(); ++i){
        auto& layer = layers[i];

        layer->set(out_stream, "   dbn->layer_get<" + std::to_string(i) + ">()");
    }

    out_stream << task_to_string("t", t) << "\n";
    out_stream << vector_to_string("actions", actions) << "\n";
    out_stream << "   dll::processor::execute(*dbn, t, actions);\n";
    out_stream << "}\n";
}

std::string command_result(const std::string& command) {
    std::stringstream output;

    char buffer[1024];

    FILE* stream = popen(command.c_str(), "r");

    if(!stream){
        return {};
    }

    while (fgets(buffer, 1024, stream) != NULL) {
        output << buffer;
    }

    if(pclose(stream)){
        return {};
    }

    std::string out(output.str());

    if(out[out.size() - 1] == '\n'){
        return {out.begin(), out.end() - 1};
    }

    return out;
}

bool compile(const char* cxx, const options& opt){
    std::cout << "Compiling the program..." << std::endl;

    std::string compile_command(cxx);

    compile_command += " -o .dbn.out ";
    compile_command += " -g ";
    compile_command += " -O2 -DETL_VECTORIZE_FULL ";
    compile_command += " -std=c++1y ";
    compile_command += " -pthread ";
    compile_command += " .dbn.cpp ";

    if(opt.mkl){
        compile_command += " -DETL_MKL_MODE ";

        auto cflags = command_result("pkg-config --cflags mkl");

        if(cflags.empty()){
            std::cout << "Failed to get compilation flags for MKL" << std::endl;
            std::cout << "   `pkg-config --cflags mkl` should return the compilation for MKL" << std::endl;
            return false;
        }

        auto ldflags = command_result("pkg-config --libs mkl");

        if(ldflags.empty()){
            std::cout << "Failed to get linking flags for MKL" << std::endl;
            std::cout << "   `pkg-config --libs mkl` should return the linking for MKL" << std::endl;
            return false;
        }

        compile_command += " " + cflags + " ";
        compile_command += " " + ldflags + " ";
    }

    int compile_result = system(compile_command.c_str());

    if(compile_result){
        std::cout << "Compilation failed" << std::endl;
        return false;
    } else {
        std::cout << "... done" << std::endl;
        return true;
    }
}

} //end of namespace dllp
