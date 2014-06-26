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
#include "fast_vector.hpp"

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

    static constexpr const std::size_t NV = Layer::NV;
    static constexpr const std::size_t NH = Layer::NH;
    static constexpr const std::size_t K = Layer::K;

    static constexpr const std::size_t NW = NV - NH + 1; //By definition

    static_assert(BatchSize > 0, "Batch size must be at least 1");

    static_assert(VisibleUnit == Type::SIGMOID,
        "Only stochastic binary units are supported");
    static_assert(HiddenUnit == Type::SIGMOID,
        "Only stochastic binary units are supported");

    //Configurable properties
    weight learning_rate = 1e-1;
    weight momentum = 0.5;

    fast_vector<fast_vector<weight, NW * NW>, K> w;     //shared weights
    fast_vector<weight, K> b;                           //hidden biases bk
    weight c;                                           //visible single bias c

    fast_vector<weight, NV * NV> v1;                    //visible units

    fast_vector<fast_vector<weight, NH * NH>, K> h1_a;  //Activation probabilities of reconstructed hidden units
    fast_vector<fast_vector<weight, NH * NH>, K> h1_s;  //Sampled values of reconstructed hidden units

    fast_vector<weight, NV * NV> v2_a;                  //Activation probabilities of reconstructed visible units
    fast_vector<weight, NV * NV> v2_s;                  //Sampled values of reconstructed visible units

    fast_vector<fast_vector<weight, NH * NH>, K> h2_a;  //Activation probabilities of reconstructed hidden units
    fast_vector<fast_vector<weight, NH * NH>, K> h2_s;  //Sampled values of reconstructed hidden units

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