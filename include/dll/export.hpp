//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file export.hpp
 * \brief Export functions to save features
 */

#ifndef DLL_EXPORT_HPP
#define DLL_EXPORT_HPP

#include <string>

namespace dll {

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

#endif
