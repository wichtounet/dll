//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file export.hpp
 * \brief Export functions to save features
 */

#pragma once

#include <string>
#include <iostream>
#include <fstream>

#include "format.hpp"

namespace dll {

/*!
 * \brief Export the given features to the given file with using DLL format
 * \param features The features to be exported
 * \param file The file into which to export the features
 */
template <typename Features>
void export_features_dll(const Features& features, const std::string& file) {
    std::ofstream os(file);

    std::string comma = "";

    for (auto& feature : features) {
        os << comma << feature;
        comma = ";";
    }

    os << '\n';
}

} //end of dll namespace
