//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <fstream>

#include "cpp_utils/assert.hpp" //Assertions
#include "cpp_utils/io.hpp"     // For binary writing

#include "etl/etl.hpp"

#include "layer.hpp"
#include "layer_traits.hpp"
#include "util/tmp.hpp"

namespace dll {

/*!
 * \brief Base class for LSTM layers (fast / dynamic)
 */
template <typename Derived, typename Desc>
struct base_lstm_layer : layer<Derived> {
    using desc      = Desc;                                    ///< The descriptor of the layer
    using derived_t = Derived;                                 ///< The derived type (CRTP)
    using weight    = typename desc::weight;                   ///< The data type for this layer
    using this_type = base_lstm_layer<derived_t, desc>; ///< The type of this layer
    using base_type = layer<Derived>;                          ///< The base type

    static constexpr auto activation_function = desc::activation_function; ///< The layer's activation function

    /*!
     * \brief Initialize the neural layer
     */
    base_lstm_layer()
            : base_type() {
        // Nothing to init here
    }

    base_lstm_layer(const base_lstm_layer& rhs) = delete;
    base_lstm_layer(base_lstm_layer&& rhs)      = delete;

    base_lstm_layer& operator=(const base_lstm_layer& rhs) = delete;
    base_lstm_layer& operator=(base_lstm_layer&& rhs) = delete;

    /*!
     * \brief Backup the weights in the secondary weights matrix
     */
    void backup_weights() {
        unique_safe_get(as_derived().bak_w_i) = as_derived().w_i;
        unique_safe_get(as_derived().bak_u_i) = as_derived().u_i;
        unique_safe_get(as_derived().bak_b_i) = as_derived().b_i;
        unique_safe_get(as_derived().bak_w_g) = as_derived().w_g;
        unique_safe_get(as_derived().bak_u_g) = as_derived().u_g;
        unique_safe_get(as_derived().bak_b_g) = as_derived().b_g;
        unique_safe_get(as_derived().bak_w_f) = as_derived().w_f;
        unique_safe_get(as_derived().bak_u_f) = as_derived().u_f;
        unique_safe_get(as_derived().bak_b_f) = as_derived().b_f;
        unique_safe_get(as_derived().bak_w_o) = as_derived().w_o;
        unique_safe_get(as_derived().bak_u_o) = as_derived().u_o;
        unique_safe_get(as_derived().bak_b_o) = as_derived().b_o;
    }

    /*!
     * \brief Restore the weights from the secondary weights matrix
     */
    void restore_weights() {
        as_derived().w_i = *as_derived().bak_w_i;
        as_derived().u_i = *as_derived().bak_u_i;
        as_derived().b_i = *as_derived().bak_b_i;
        as_derived().w_g = *as_derived().bak_w_g;
        as_derived().u_g = *as_derived().bak_u_g;
        as_derived().b_g = *as_derived().bak_b_g;
        as_derived().w_f = *as_derived().bak_w_f;
        as_derived().u_f = *as_derived().bak_u_f;
        as_derived().b_f = *as_derived().bak_b_f;
        as_derived().w_o = *as_derived().bak_w_o;
        as_derived().u_o = *as_derived().bak_u_o;
        as_derived().b_o = *as_derived().bak_b_o;
    }

    /*!
     * \brief Load the weigts into the given stream
     */
    void store(std::ostream& os) const {
        cpp::binary_write_all(os, as_derived().w_i);
        cpp::binary_write_all(os, as_derived().u_i);
        cpp::binary_write_all(os, as_derived().b_i);
        cpp::binary_write_all(os, as_derived().w_g);
        cpp::binary_write_all(os, as_derived().u_g);
        cpp::binary_write_all(os, as_derived().b_g);
        cpp::binary_write_all(os, as_derived().w_f);
        cpp::binary_write_all(os, as_derived().u_f);
        cpp::binary_write_all(os, as_derived().b_f);
        cpp::binary_write_all(os, as_derived().w_o);
        cpp::binary_write_all(os, as_derived().u_o);
        cpp::binary_write_all(os, as_derived().b_o);
    }

    /*!
     * \brief Load the weigts from the given stream
     */
    void load(std::istream& is) {
        cpp::binary_load_all(is, as_derived().w_i);
        cpp::binary_load_all(is, as_derived().u_i);
        cpp::binary_load_all(is, as_derived().b_i);
        cpp::binary_load_all(is, as_derived().w_g);
        cpp::binary_load_all(is, as_derived().u_g);
        cpp::binary_load_all(is, as_derived().b_g);
        cpp::binary_load_all(is, as_derived().w_f);
        cpp::binary_load_all(is, as_derived().u_f);
        cpp::binary_load_all(is, as_derived().b_f);
        cpp::binary_load_all(is, as_derived().w_o);
        cpp::binary_load_all(is, as_derived().u_o);
        cpp::binary_load_all(is, as_derived().b_o);
    }

    /*!
     * \brief Load the weigts into the given file
     */
    void store(const std::string& file) const {
        std::ofstream os(file, std::ofstream::binary);
        store(os);
    }

    /*!
     * \brief Load the weigts from the given file
     */
    void load(const std::string& file) {
        std::ifstream is(file, std::ifstream::binary);
        load(is);
    }

    /*!
     * \brief Returns the trainable variables of this layer.
     * \return a tuple containing references to the variables of this layer
     */
    decltype(auto) trainable_parameters() {
        return std::make_tuple(
            std::ref(as_derived().w_i), std::ref(as_derived().u_i), std::ref(as_derived().b_i),
            std::ref(as_derived().w_g), std::ref(as_derived().u_g), std::ref(as_derived().b_g),
            std::ref(as_derived().w_f), std::ref(as_derived().u_f), std::ref(as_derived().b_f),
            std::ref(as_derived().w_o), std::ref(as_derived().u_o), std::ref(as_derived().b_o));
    }

    /*!
     * \brief Returns the trainable variables of this layer.
     * \return a tuple containing references to the variables of this layer
     */
    decltype(auto) trainable_parameters() const {
        return std::make_tuple(
            std::cref(as_derived().w_i), std::cref(as_derived().u_i), std::cref(as_derived().b_i),
            std::cref(as_derived().w_g), std::cref(as_derived().u_g), std::cref(as_derived().b_g),
            std::cref(as_derived().w_f), std::cref(as_derived().u_f), std::cref(as_derived().b_f),
            std::cref(as_derived().w_o), std::cref(as_derived().u_o), std::cref(as_derived().b_o));
    }

private:
    //CRTP Deduction

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() {
        return *static_cast<derived_t*>(this);
    }

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const derived_t& as_derived() const {
        return *static_cast<const derived_t*>(this);
    }
};

} //end of dll namespace
