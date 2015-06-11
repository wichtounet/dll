//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_TRAITS_HPP
#define DLL_DBN_TRAITS_HPP

#include "tmp.hpp"
#include "decay_type.hpp"

namespace dll {

/*!
 * \brief Type Traits to get information on DBN type
 */
template<typename DBN>
struct dbn_traits {
    using dbn_t = DBN;
    using desc = typename dbn_t::desc;

    /*!
     * \brief Indicates if the DBN is convolutional
     */
    static constexpr bool is_convolutional() noexcept {
        return desc::layers::is_convolutional;
    }

    /*!
     * \brief Indicates if the DBN is multiplex
     */
    static constexpr bool is_multiplex() noexcept {
        return desc::layers::is_multiplex;
    }

    /*!
     * \brief Indicates if the DBN is dynamic
     */
    static constexpr bool is_dynamic() noexcept {
        return desc::layers::is_dynamic;
    }

    static constexpr bool has_momentum() noexcept {
        return desc::parameters::template contains<momentum>();
    }

    static constexpr bool save_memory() noexcept {
        return desc::parameters::template contains<memory>();
    }

    static constexpr bool concatenate() noexcept {
        return desc::parameters::template contains<svm_concatenate>();
    }

    static constexpr bool is_parallel() noexcept {
        return desc::parameters::template contains<parallel>();
    }

    static constexpr bool is_verbose() noexcept {
        return desc::parameters::template contains<verbose>();
    }

    static constexpr bool scale() noexcept {
        return desc::parameters::template contains<svm_scale>();
    }

    static constexpr decay_type decay() noexcept {
        return detail::get_value_l<weight_decay<decay_type::NONE>, typename desc::parameters>::value;
    }
};

/** Functions to get the dimensions of DBN regardless of dynamic or not **/

template<typename DBN, cpp::disable_if_u<dbn_traits<DBN>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t dbn_output_size(const DBN& /*dbn*/){
    return DBN::output_size();
}

template<typename DBN, cpp::enable_if_u<dbn_traits<DBN>::is_dynamic()> = cpp::detail::dummy>
std::size_t dbn_output_size(const DBN& dbn){
    return dbn.output_size();
}

template<typename DBN, cpp::disable_if_u<dbn_traits<DBN>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t dbn_full_output_size(const DBN& /*dbn*/){
    return DBN::full_output_size();
}

template<typename DBN, cpp::enable_if_u<dbn_traits<DBN>::is_dynamic()> = cpp::detail::dummy>
std::size_t dbn_full_output_size(const DBN& dbn){
    return dbn.full_output_size();
}

template<typename DBN, cpp::disable_if_u<dbn_traits<DBN>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t dbn_input_size(const DBN& /*dbn*/){
    return DBN::input_size();
}

template<typename DBN, cpp::enable_if_u<dbn_traits<DBN>::is_dynamic()> = cpp::detail::dummy>
std::size_t dbn_input_size(const DBN& dbn){
    return dbn.input_size();
}

template<typename DBN, cpp::disable_if_u<dbn_traits<DBN>::is_dynamic()> = cpp::detail::dummy>
constexpr std::size_t dbn_full_input_size(const DBN& /*dbn*/){
    return DBN::full_input_size();
}

template<typename DBN, cpp::enable_if_u<dbn_traits<DBN>::is_dynamic()> = cpp::detail::dummy>
std::size_t dbn_full_input_size(const DBN& dbn){
    return dbn.full_input_size();
}

} //end of dll namespace

#endif
