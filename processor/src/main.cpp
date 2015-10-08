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

using options = dll::processor::options;

struct layer {
    virtual void print(std::ostream& out) const = 0;
    virtual std::size_t hidden_get() const = 0;

    virtual bool is_conv() const { return false; }

    virtual std::size_t hidden_get_1() const { return 0; }
    virtual std::size_t hidden_get_2() const { return 0; }
    virtual std::size_t hidden_get_3() const { return 0; }

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

std::string activation_function(const std::string& unit){
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

std::string decay_to_str(const std::string& decay){
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

bool valid_unit(const std::string& unit){
    return unit == "binary" || unit == "softmax" || unit == "gaussian";
}

bool valid_activation(const std::string& unit){
    return unit == "sigmoid" || unit == "softmax" || unit == "tanh" || unit == "relu";
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

struct dense_layer : layer {
    std::size_t visible = 0;
    std::size_t hidden = 0;

    std::string activation;

    void print(std::ostream& out) const override {
        out << "dll::dense_desc<" << visible << ", " << hidden;

        if(!activation.empty()){
            out << ", dll::activation<dll::function::" << activation_function(activation) << ">";
        }

        out << ">::layer_t";
    }

    virtual void set(std::ostream& /*out*/, const std::string& /*lhs*/) const override {
        //Nothing to set here
    }

    std::size_t hidden_get() const override {
        return hidden;
    }
};

struct conv_layer : layer {
    std::size_t c = 0;
    std::size_t v1 = 0;
    std::size_t v2 = 0;
    std::size_t k = 0;
    std::size_t w1 = 0;
    std::size_t w2 = 0;

    std::string activation;

    double learning_rate = dll::processor::stupid_default;
    double momentum = dll::processor::stupid_default;

    bool is_conv() const override {
        return false;
    }

    void print(std::ostream& out) const override {
        out << "dll::conv_desc<" << c << ", " << v1 << ", " << v2 << ", " << k << ", " << (v1 - w1 + 1) << ", " << (v2 - w2 + 1);

        if(!activation.empty()){
            out << ", dll::activation<dll::function::" << activation_function(activation) << ">";
        }

        out << ">::layer_t";
    }

    virtual void set(std::ostream& /*out*/, const std::string& /*lhs*/) const override {
        //Nothing to set here
    }

    std::size_t hidden_get() const override {
        return k * (v1 - w1 + 1) * (v2 - w2 + 1);
    }

    std::size_t hidden_get_1() const override {
        return k;
    }

    std::size_t hidden_get_2() const override {
        return v1 - w1 + 1;
    }

    std::size_t hidden_get_3() const override {
        return v2 - w2 + 1;
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
        } else if(starts_with(lines[i], "scale: ")){
            source.scale = true;
            source.scale_d = std::stod(extract_value(lines[i], "scale: "));
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


void generate(const std::vector<std::shared_ptr<dllp::layer>>& layers, const dll::processor::task& t, const std::vector<std::string>& actions);
bool compile(const options& opt);

void parse_options(int argc, char* argv[], dllp::options& opt, std::vector<std::string>& actions, std::string& source_file){
    std::size_t i = 1;

    while(true){
        if(std::string(argv[i]) == "--mkl"){
            opt.mkl = true;
            ++i;
        } else if(std::string(argv[i]) == "--cufft"){
            opt.cufft = true;
            ++i;
        } else if(std::string(argv[i]) == "--cublas"){
            opt.cublas = true;
            ++i;
        } else if(std::string(argv[i]) == "--cache"){
            opt.cache = true;
            ++i;
        } else {
            break;
        }
    }

    source_file = argv[i++];

    for(; i < std::size_t(argc); ++i){
        actions.emplace_back(argv[i]);
    }
}

bool parse_file(const std::string& source_file, dll::processor::task& t, std::vector<std::shared_ptr<dllp::layer>>& layers){
    //Parse the source_file

    std::ifstream source_stream(source_file);

    std::string current_line;
    std::vector<std::string> lines;

    while(std::getline(source_stream, current_line)) {
        std::string processed(cpp::trim(current_line));

        if(!processed.empty()){
            lines.emplace_back(std::move(processed));
        }
    }

    if(lines.empty()){
        std::cout << "dllp: error: file does not exist or is empty" << std::endl;
        return false;
    }

    //Process the lines

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
                                return false;
                            }
                        } else if(dllp::starts_with(lines[i], "visible_unit:")){
                            rbm->visible_unit = dllp::extract_value(lines[i], "visible_unit: ");
                            ++i;

                            if(!dllp::valid_unit(rbm->visible_unit)){
                                std::cout << "dllp: error: invalid visible unit type must be one of [binary, softmax, gaussian]" << std::endl;
                                return false;
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
                } else if(lines[i] == "dense:"){
                    ++i;

                    auto dense = std::make_shared<dllp::dense_layer>();

                    while(i < lines.size()){
                        if(dllp::starts_with(lines[i], "visible:")){
                            dense->visible = std::stol(dllp::extract_value(lines[i], "visible: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "hidden:")){
                            dense->hidden = std::stol(dllp::extract_value(lines[i], "hidden: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "activation:")){
                            dense->activation = dllp::extract_value(lines[i], "activation: ");
                            ++i;

                            if(!dllp::valid_activation(dense->activation)){
                                std::cout << "dllp: error: invalid activation function, must be [sigmoid,tanh,relu,softmax]" << std::endl;
                                return false;
                            }
                        } else {
                            break;
                        }
                    }

                    if(layers.empty() && (dense->visible == 0 || dense->hidden == 0)){
                        std::cout << "dllp: error: The first layer needs visible and hidden sizes" << std::endl;
                    } else if(!layers.empty() && dense->hidden == 0){
                        std::cout << "dllp: error: The number of hidden units is mandatory" << std::endl;
                    }

                    if(!layers.empty()){
                        dense->visible = layers.back()->hidden_get();
                    }

                    layers.push_back(std::move(dense));
                } else if(lines[i] == "conv:"){
                    ++i;

                    auto conv = std::make_shared<dllp::conv_layer>();

                    while(i < lines.size()){
                        if(dllp::starts_with(lines[i], "channels:")){
                            conv->c = std::stol(dllp::extract_value(lines[i], "channels: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "filters:")){
                            conv->k = std::stol(dllp::extract_value(lines[i], "filters: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "v1:")){
                            conv->v1 = std::stol(dllp::extract_value(lines[i], "v1: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "v2:")){
                            conv->v2 = std::stol(dllp::extract_value(lines[i], "v2: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "w1:")){
                            conv->w1 = std::stol(dllp::extract_value(lines[i], "w1: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "w2:")){
                            conv->w2 = std::stol(dllp::extract_value(lines[i], "w2: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "activation:")){
                            conv->activation = dllp::extract_value(lines[i], "activation: ");
                            ++i;

                            if(!dllp::valid_activation(conv->activation)){
                                std::cout << "dllp: error: invalid activation function, must be [sigmoid,tanh,relu,softmax]" << std::endl;
                                return false;
                            }
                        } else {
                            break;
                        }
                    }

                    if(layers.empty() && (!conv->c || !conv->v1 || !conv->v2 || !conv->k || !conv->w1 || !conv->w2)){
                        std::cout << "dllp: error: The first layer needs input and output sizes" << std::endl;
                    } else if(!layers.empty() && !conv->k){
                        std::cout << "dllp: error: The number of filters is mandatory" << std::endl;
                    } else if(!layers.empty() && (!conv->w1 || !conv->w2)){
                        std::cout << "dllp: error: The size of the filters is mandatory" << std::endl;
                    }

                    if(!layers.empty()){
                        conv->c = layers.back()->hidden_get_1();
                        conv->v1 = layers.back()->hidden_get_2();
                        conv->v2 = layers.back()->hidden_get_3();
                    }

                    layers.push_back(std::move(conv));
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
                        } else if(dllp::starts_with(lines[i], "batch:")){
                            t.ft_desc.batch_size = std::stol(dllp::extract_value(lines[i], "batch: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "weight_decay:")){
                            t.ft_desc.decay = dllp::extract_value(lines[i], "weight_decay: ");
                            ++i;
                        } else if(dllp::starts_with(lines[i], "l1_weight_cost:")){
                            t.ft_desc.l1_weight_cost = std::stod(dllp::extract_value(lines[i], "l1_weight_cost: "));
                            ++i;
                        } else if(dllp::starts_with(lines[i], "l2_weight_cost:")){
                            t.ft_desc.l2_weight_cost = std::stod(dllp::extract_value(lines[i], "l2_weight_cost: "));
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

            return false;
        }
    }

    if(layers.empty()){
        std::cout << "dllp: error: no layer has been declared" << std::endl;
        return false;
    }

    return true;
}

bool compile_exe(const dllp::options& opt, const std::vector<std::string>& actions, const std::string& source_file, const dll::processor::task& t, const std::vector<std::shared_ptr<dllp::layer>>& layers){
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
        if(!dllp::compile(opt)){
            return false;
        }
    }

    return true;
}

int process_file(const dllp::options& opt, const std::vector<std::string>& actions, const std::string& source_file){
    //1. Parse the configuration file

    dll::processor::task t;
    std::vector<std::shared_ptr<dllp::layer>> layers;

    if(!parse_file(source_file, t, layers)){
        return 1;
    }

    //2. Generate the executable

    if(!compile_exe(opt, actions, source_file, t, layers)){
        return 1;
    }

    //3. Run the generated program

    std::cout << "Executing the program" << std::endl;

    auto exec_result = system("./.dbn.out");

    if(exec_result){
        std::cout << "Impossible to execute the generated file" << std::endl;
        return exec_result;
    }

    return 0;
}

} //end of dllp namespace

int main(int argc, char* argv[]){
    if(argc < 3){
        std::cout << "dllp: Not enough arguments" << std::endl;
        dllp::print_usage();
        return 1;
    }

    //Check that $CXX is defined

    const auto* cxx = std::getenv("CXX");

    if(!cxx){
        std::cout << "CXX environment variable must be set" << std::endl;
        return 2;
    }

    //Parse the options

    dllp::options opt;
    std::vector<std::string> actions;
    std::string source_file;

    dllp::parse_options(argc, argv, opt, actions, source_file);

    //Process the file

    return dllp::process_file(opt, actions, source_file);
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

void generate(const std::vector<std::shared_ptr<dllp::layer>>& layers, const dll::processor::task& t, const std::vector<std::string>& actions){
    std::ofstream out_stream(".dbn.cpp");

    out_stream << "#include <memory>\n";
    out_stream << "#include \"dll/processor/processor.hpp\"\n";
    out_stream << "#include \"dll/stochastic_gradient_descent.hpp\"\n\n";

    out_stream << "using dbn_t = dll::dbn_desc<dll::dbn_layers<\n";

    std::string comma = "  ";

    for(auto& layer : layers){
        out_stream << comma;
        layer->print(out_stream);
        comma = "\n, ";
    }

    out_stream << "\n>";

    out_stream << ", dll::trainer<dll::sgd_trainer>\n";

    if(t.ft_desc.momentum != dll::processor::stupid_default){
        out_stream << ", dll::momentum\n";
    }

    if(t.ft_desc.batch_size > 0){
        out_stream << ", dll::batch_size<" << t.ft_desc.batch_size << ">\n";
    }

    out_stream << ", dll::weight_decay<dll::decay_type::" << decay_to_str(t.ft_desc.decay) << ">\n";

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

    if(t.ft_desc.l1_weight_cost != dll::processor::stupid_default){
        out_stream << "   dbn->l1_weight_cost = " << t.ft_desc.l1_weight_cost << ";\n";
    }

    if(t.ft_desc.l2_weight_cost != dll::processor::stupid_default){
        out_stream << "   dbn->l2_weight_cost = " << t.ft_desc.l2_weight_cost << ";\n";
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

bool append_pkg_flags(std::string& flags, const std::string& pkg){
    auto cflags = command_result("pkg-config --cflags " + pkg);

    if(cflags.empty()){
        std::cout << "Failed to get compilation flags for " << pkg << std::endl;
        std::cout << "   `pkg-config --cflags " << pkg << "` should return the compilation for " << pkg << std::endl;
        return false;
    }

    auto ldflags = command_result("pkg-config --libs " + pkg);

    if(ldflags.empty()){
        std::cout << "Failed to get linking flags for " << pkg << std::endl;
        std::cout << "   `pkg-config --libs " << pkg << "` should return the linking for " << pkg << std::endl;
        return false;
    }

    flags += " " + cflags + " ";
    flags += " " + ldflags + " ";

    return true;
}

bool compile(const options& opt){
    std::cout << "Compiling the program..." << std::endl;

    const auto* cxx = std::getenv("CXX");

    std::string compile_command(cxx);

    compile_command += " -o .dbn.out ";
    compile_command += " -g ";
    compile_command += " -O2 -DETL_VECTORIZE_FULL ";
    compile_command += " -std=c++1y ";
    compile_command += " -pthread ";
    compile_command += " .dbn.cpp ";

    if(opt.mkl){
        compile_command += " -DETL_MKL_MODE ";

        if(!append_pkg_flags(compile_command, "mkl")){
            return false;
        }
    }

    if(opt.cublas){
        compile_command += " -DETL_CUBLAS_MODE ";

        if(!append_pkg_flags(compile_command, "cublas")){
            return false;
        }
    }

    if(opt.cufft){
        compile_command += " -DETL_CUFFT_MODE ";

        if(!append_pkg_flags(compile_command, "cufft")){
            return false;
        }
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

std::string dll::processor::process_file_result(const dllp::options& opt, const std::vector<std::string>& actions, const std::string& source_file){
    //1. Parse the configuration file

    dll::processor::task t;
    std::vector<std::shared_ptr<dllp::layer>> layers;

    if(!dllp::parse_file(source_file, t, layers)){
        return "";
    }

    //2. Generate the executable

    if(!dllp::compile_exe(opt, actions, source_file, t, layers)){
        return "";
    }

    //3. Execute and return the result directly

    return dllp::command_result("./.dbn.out");
}

