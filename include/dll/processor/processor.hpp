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

//TODO

} //end of namespace processor

} //end of namespace dll
