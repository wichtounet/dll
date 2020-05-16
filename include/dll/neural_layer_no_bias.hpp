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
#include "util/tmp.hpp"
#include "layer_traits.hpp"

namespace dll {

/*!
 * \brief Standard dense layer of neural network without bias.
 */
template <typename Derived, typename Desc>
struct neural_layer_no_bias : layer<Derived> {
    using desc      = Desc;                                  ///< The descriptor of the layer
    using derived_t = Derived;                               ///< The derived type (CRTP)
    using weight    = typename desc::weight;                 ///< The data type for this layer
    using this_type = neural_layer_no_bias<derived_t, desc>; ///< The type of this layer
    using base_type = layer<Derived>;                        ///< The base type

    /*!
     * \brief Initialize the neural layer
     */
    neural_layer_no_bias() : base_type() {
        // Nothing to init here
    }

    neural_layer_no_bias(neural_layer_no_bias& rhs) = delete;
    neural_layer_no_bias(neural_layer_no_bias&& rhs) = delete;

    neural_layer_no_bias& operator=(neural_layer_no_bias& rhs) = delete;
    neural_layer_no_bias& operator=(neural_layer_no_bias&& rhs) = delete;

    /*!
     * \brief Backup the weights in the secondary weights matrix
     */
    void backup_weights() {
        unique_safe_get(as_derived().bak_w) = as_derived().w;
    }

    /*!
     * \brief Restore the weights from the secondary weights matrix
     */
    void restore_weights() {
        as_derived().w = *as_derived().bak_w;
    }

    /*!
     * \brief Load the weigts into the given stream
     */
    void store(std::ostream& os) const {
        cpp::binary_write_all(os, as_derived().w);
    }

    /*!
     * \brief Load the weigts from the given stream
     */
    void load(std::istream& is) {
        cpp::binary_load_all(is, as_derived().w);
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
    decltype(auto) trainable_parameters(){
        return std::make_tuple(std::ref(as_derived().w));
    }

    /*!
     * \brief Returns the trainable variables of this layer.
     * \return a tuple containing references to the variables of this layer
     */
    decltype(auto) trainable_parameters() const {
        return std::make_tuple(std::cref(as_derived().w));
    }

private:
    //CRTP Deduction

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived(){
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
