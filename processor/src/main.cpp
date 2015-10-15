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

namespace {

void print_usage() {
    std::cout << "Usage: dllp conf_file action" << std::endl;
}

void parse_options(int argc, char* argv[], dll::processor::options& opt, std::vector<std::string>& actions, std::string& source_file) {
    std::size_t i = 1;

    while (true) {
        if (std::string(argv[i]) == "--mkl") {
            opt.mkl = true;
            ++i;
        } else if (std::string(argv[i]) == "--cufft") {
            opt.cufft = true;
            ++i;
        } else if (std::string(argv[i]) == "--cublas") {
            opt.cublas = true;
            ++i;
        } else if (std::string(argv[i]) == "--cache") {
            opt.cache = true;
            ++i;
        } else {
            break;
        }
    }

    source_file = argv[i++];

    for (; i < std::size_t(argc); ++i) {
        actions.emplace_back(argv[i]);
    }
}

} //end of anonymous namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "dllp: Not enough arguments" << std::endl;
        print_usage();
        return 1;
    }

    //Check that $CXX is defined

    const auto* cxx = std::getenv("CXX");

    if (!cxx) {
        std::cout << "CXX environment variable must be set" << std::endl;
        return 2;
    }

    //Parse the options

    dll::processor::options opt;
    std::vector<std::string> actions;
    std::string source_file;

    parse_options(argc, argv, opt, actions, source_file);

    //Process the file

    return dll::processor::process_file(opt, actions, source_file);
}
