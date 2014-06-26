//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONV_RBM_HPP
#define DBN_CONV_RBM_HPP

#include <cstddef>

#include "unit_type.hpp"

namespace dbn {

/*!
 * \brief Convolutioal Restricted Boltzmann Machine
 */
template<typename Layer>
class conv_rbm {
public:
    typedef double weight;
    typedef double value_t;

    static constexpr const bool Momentum = Layer::Conf::Momentum;
    static constexpr const std::size_t BatchSize = Layer::Conf::BatchSize;
    static constexpr const Type VisibleUnit = Layer::Conf::VisibleUnit;
    static constexpr const Type HiddenUnit = Layer::Conf::HiddenUnit;

    static_assert(BatchSize > 0, "Batch size must be at least 1");

    static_assert(VisibleUnit == Type::SIGMOID,
        "Only stochastic binary units are supported");
    static_assert(HiddenUnit == Type::SIGMOID,
        "Only stochastic binary units are supported");

    //Configurable properties
    weight learning_rate = 1e-1;
    weight momentum = 0.5;

public:
    //No copying
    conv_rbm(const conv_rbm& rbm) = delete;
    conv_rbm& operator=(const conv_rbm& rbm) = delete;

    //No moving
    conv_rbm(conv_rbm&& rbm) = delete;
    conv_rbm& operator=(conv_rbm&& rbm) = delete;

    conv_rbm(){}
};

} //end of dbn namespace

#endif