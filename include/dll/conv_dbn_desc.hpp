//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_CONV_DBN_DESC_HPP
#define DLL_CONV_DBN_DESC_HPP

#include "base_conf.hpp"
#include "watcher.hpp"
#include "tmp.hpp"

namespace dll {

/*!
 * \brief Describe a Convolutional DBN *
 *
 * This struct should be used to define a DBN.
 * Once configured, the ::dbn_t member returns the type of the configured DBN.
 */
template<typename Layers, typename... Parameters>
struct conv_dbn_desc {
    using layers = Layers;

    /*! The type of the watched to use during training */
    template <typename DBN>
    using watcher_t = typename detail::get_template_type<watcher<default_dbn_watcher>, Parameters...>::template type<DBN>;

    /*! The DBN type */
    using dbn_t = conv_dbn<conv_dbn_desc<Layers, Parameters...>>;

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<detail::tmp_list<watcher_id>, Parameters...>::value,
        "Invalid parameters type");
};

} //end of dbn namespace

#endif