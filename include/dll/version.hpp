//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

/*!
 * \brief The current major version number of the library
 */
constexpr size_t version_major = 1;

/*!
 * \brief The current minor version number of the library
 */
constexpr size_t version_minor = 0;

/*!
 * \brief The current revision version number of the library
 */
constexpr size_t version_revision = 0;

/*!
 * \brief The current version number of the library in string form.
 */
constexpr const char* version_str = "1.0";

} //end of dll namespace

/*!
 * \brief String representation of the current version of the library.
 */
#define DLL_VERSION_STR "1.0"

/*!
 * \brief The current major version number of the library
 */
#define DLL_VERSION_MAJOR 1

/*!
 * \brief The current minor version number of the library
 */
#define DLL_VERSION_MINOR 0

/*!
 * \brief The current revision version number of the library
 */
#define DLL_VERSION_REVISION 0
