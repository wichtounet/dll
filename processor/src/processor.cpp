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

#include "parse_utils.hpp"
#include "layer.hpp"

#include "dll/processor/processor.hpp"

namespace dllp {

using options = dll::processor::options;

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

bool parse_file(const std::string& source_file, dll::processor::task& t, std::vector<std::shared_ptr<dllp::layer>>& layers){
    //0. Parse the source file

    auto lines = read_lines(source_file);

    if(lines.empty()){
        std::cout << "dllp: warning: included file is empty or does not exist" << std::endl;
        return false;
    }

    //1. Process includes

    for(std::size_t i = 0; i < lines.size();){
        auto& current_line = lines[i];

        if(dllp::starts_with(current_line, "include: ")){
            auto include_file = dllp::extract_value(current_line, "include: ");
            auto include_lines = read_lines(include_file);

            if(lines.empty()){
                std::cout << "dllp: error: file does not exist or is empty" << std::endl;
            } else {
                std::copy(include_lines.begin(), include_lines.end(), std::inserter(lines, lines.begin() + i));

                i += include_lines.size();
            }
        }

        ++i;
    }


    //2. Process the lines

    for(std::size_t i = 0; i < lines.size();){
        auto& current_line = lines[i];

        if(dllp::starts_with(current_line, "include: ")){
            ++i;
        } else if(current_line == "data:"){
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

                    if(!rbm->parse(layers, lines, i)){
                        return false;
                    }

                    layers.push_back(std::move(rbm));
                } else if(lines[i] == "crbm:"){
                    ++i;

                    auto crbm = std::make_shared<dllp::conv_rbm_layer>();

                    if(!crbm->parse(layers, lines, i)){
                        return false;
                    }

                    layers.push_back(std::move(crbm));
                } else if(lines[i] == "dense:"){
                    ++i;

                    auto dense = std::make_shared<dllp::dense_layer>();

                    if(!dense->parse(layers, lines, i)){
                        return false;
                    }

                    layers.push_back(std::move(dense));
                } else if(lines[i] == "conv:"){
                    ++i;

                    auto conv = std::make_shared<dllp::conv_layer>();

                    if(!conv->parse(layers, lines, i)){
                        return false;
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
                    if(!opt.quiet){
                        std::cout << "Skip compilation" << std::endl;
                    }
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
    if(!opt.quiet){
        std::cout << "Compiling the program..." << std::endl;
    }

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
        if(!opt.quiet){
            std::cout << "... done" << std::endl;
        }
        return true;
    }
}

} //end of namespace dllp

int dll::processor::process_file(const dllp::options& opt, const std::vector<std::string>& actions, const std::string& source_file){
    //1. Parse the configuration file

    dll::processor::task t;
    std::vector<std::shared_ptr<dllp::layer>> layers;

    if(!dllp::parse_file(source_file, t, layers)){
        return 1;
    }

    //2. Generate the executable

    if(!dllp::compile_exe(opt, actions, source_file, t, layers)){
        return 1;
    }

    //3. Run the generated program

    if(!opt.quiet){
        std::cout << "Executing the program" << std::endl;
    }

    auto exec_result = system("./.dbn.out");

    if(exec_result){
        std::cout << "Impossible to execute the generated file" << std::endl;
        return exec_result;
    }

    return 0;
}

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
