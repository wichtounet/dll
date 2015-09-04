//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_DESC_HPP
#define DLL_DBN_DESC_HPP

#include "base_conf.hpp"
#include "conjugate_gradient.hpp"
#include "watcher.hpp"
#include "tmp.hpp"

namespace dll {

template <typename DBN>
using default_dbn_trainer_t = cg_trainer<DBN, false>;

/*!
 * \brief Describe a DBN *
 *
 * This struct should be used to define a DBN.
 * Once configured, the ::dbn_t member returns the type of the configured DBN.
 */
template<typename Layers, typename... Parameters>
struct dbn_desc {
    using layers = Layers;
    using parameters = cpp::type_list<Parameters...>;

    static constexpr const std::size_t BatchSize = detail::get_value<batch_size<1>, Parameters...>::value;
    static constexpr const std::size_t BigBatchSize = detail::get_value<big_batch_size<1>, Parameters...>::value;

    /*! The type of the trainer to use to train the DBN */
    template <typename DBN>
    using trainer_t = typename detail::get_template_type<trainer<default_dbn_trainer_t>, Parameters...>::template value<DBN>;

    /*! The type of the watched to use during training */
    template <typename DBN>
    using watcher_t = typename detail::get_template_type<watcher<default_dbn_watcher>, Parameters...>::template value<DBN>;

    /*! The DBN type */
    using dbn_t = dbn<dbn_desc<Layers, Parameters...>>;

    static_assert(BatchSize > 0, "Batch size must be at least 1");
    static_assert(BigBatchSize > 0, "Big Batch size must be at least 1");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<
            cpp::type_list<
                trainer_id, watcher_id, momentum_id, weight_decay_id, big_batch_size_id, batch_size_id, verbose_id,
                memory_id, svm_concatenate_id, svm_scale_id, serial_id>,
            Parameters...>::value,
        "Invalid parameters type");
};

} //end of dll namespace

#endif
