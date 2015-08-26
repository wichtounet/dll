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

#include "cpp_utils/string.hpp"

namespace {

struct datasource {
    std::string source_file;
    std::string reader;
};

struct task {
    datasource pretraining;
    datasource samples;
    datasource labels;
};

void print_usage(){
    std::cout << "Usage: dll conf_file action" << std::endl;
}

bool starts_with(const std::string& str, const std::string& search){
    return std::mismatch(search.begin(), search.end(), str.begin()).first == search.end();
}

std::string extract_value(const std::string& str, const std::string& search){
    return {str.begin() + str.find(search), str.end()};
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
        std::cout << "dll:: error: missing source" << std::endl;
    }

    return source;
}

} //end of anonymous namespace

int main(int argc, char* argv[]){
    if(argc < 3){
        std::cout << "dll: Not enough arguments" << std::endl;
        print_usage();
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

    task t;

    for(std::size_t i = 0; i < lines.size(); ++i){
        auto& current_line = lines[i];

        if(current_line == "input:"){
            ++i;

            if(i == lines.size()){
                std::cout << "dll: error: input expect at least one child" << std::endl;

                return 1;
            }

            while(i < lines.size()){
                if(lines[i] == "pretraining:"){
                    t.pretraining = parse_datasource(lines, ++i);
                } else if(lines[i] == "samples:"){
                    t.samples = parse_datasource(lines, ++i);
                } else if(lines[i] == "labels:"){
                    t.labels = parse_datasource(lines, ++i);
                } else {
                    break;
                }
            }
        } else if(current_line == "rbm"){

        } else {
            std::cout << "dll: error: Invalid line: " << current_line << std::endl;

            return 1;
        }
    }

    return 0;
}
