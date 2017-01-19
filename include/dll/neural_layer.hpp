//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/assert.hpp" //Assertions

#include "etl/etl.hpp"

#include "layer.hpp"
#include "util/tmp.hpp"
#include "util/converter.hpp"
#include "layer_traits.hpp"

namespace dll {

/*!
 * \brief Standard dense layer of neural network.
 */
template <typename Derived, typename Desc>
struct neural_layer : layer<Derived> {
    using desc      = Desc;
    using derived_t = Derived;
    using weight    = typename desc::weight;
    using this_type = neural_layer<derived_t, desc>;
    using base_type = layer<Derived>;

    neural_layer() : base_type() {
        // Nothing to init here
    }

    void backup_weights() {
        unique_safe_get(as_derived().bak_w) = as_derived().w;
        unique_safe_get(as_derived().bak_b) = as_derived().b;
    }

    void restore_weights() {
        as_derived().w = *as_derived().bak_w;
        as_derived().b = *as_derived().bak_b;
    }

private:
    //CRTP Deduction

    derived_t& as_derived(){
        return *static_cast<derived_t*>(this);
    }

    const derived_t& as_derived() const {
        return *static_cast<const derived_t*>(this);
    }
};

} //end of dll namespace
