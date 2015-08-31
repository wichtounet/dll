//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*!
 * \file processor.hpp
 * \brief This file is made to be included by the dllp generated file only.
 */

#include <string>

#include "dll/dbn.hpp"

namespace dll {

namespace processor {

struct datasource {
    std::string source_file;
    std::string reader;

    datasource(){}
    datasource(std::string source_file, std::string reader) : source_file(source_file), reader(reader) {}
};

using datasource_p = const std::unique_ptr<datasource>;

template<typename DBN>
void execute(DBN& dbn, datasource_p& pt, datasource_p& fts, datasource_p& ftl, const std::vector<std::string>& actions){
    std::cout << "Configured network:" << std::endl;
    dbn.display();

    //Execute all the actions sequentially
    for(auto& action : actions){
        if(action == "pretrain"){
            if(!pt){
                std::cout << "dllp: error: pretrain is not possible with a pretraining input" << std::endl;
                return;
            }
        } else if(action == "train"){
            if(!fts || !ftl){
                std::cout << "dllp: error: train is not possible without samples and labels" << std::endl;
                return;
            }

        } else if(action == "test"){
            if(!fts || !ftl){
                std::cout << "dllp: error: test is not possible without samples and labels" << std::endl;
                return;
            }

        } else {
            std::cout << "dllp: error: Invalid action: " << action << std::endl;
        }
    }


    //TODO
}

} //end of namespace processor

} //end of namespace dll
