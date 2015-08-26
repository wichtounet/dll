//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include<string>
#include<iostream>

namespace {

void print_usage(){
    std::cout << "Usage: dll conf_file action" << std::endl;
}

} //end of anonymous namespace

int main(int argc, char* argv[]){
    if(argc < 3){
        std::cout << "dll: Not enough arguments" << std::endl;
        print_usage();
        return 0;
    }

    std::string souce_file(argv[1]);
    std::string action(argv[2]);

    //TODO

    return 0;
}
