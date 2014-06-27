//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_CONV_RBM_HPP
#define DBN_CONV_RBM_HPP

#include <cstddef>

#include "etl/fast_vector.hpp"

#include "unit_type.hpp"

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine
 */
template<typename Layer>
class conv_rbm {
public:
    typedef double weight;
    typedef double value_t;

    static constexpr const bool Momentum = Layer::Momentum;
    static constexpr const std::size_t BatchSize = Layer::BatchSize;
    static constexpr const Type VisibleUnit = Layer::VisibleUnit;
    static constexpr const Type HiddenUnit = Layer::HiddenUnit;

    static constexpr const std::size_t NV = Layer::NV;
    static constexpr const std::size_t NH = Layer::NH;
    static constexpr const std::size_t K = Layer::K;

    static constexpr const std::size_t NW = NV - NH + 1; //By definition

    static_assert(BatchSize > 0, "Batch size must be at least 1");

    static_assert(VisibleUnit == Type::SIGMOID, "Only binary visible units are supported");
    static_assert(HiddenUnit == Type::SIGMOID, "Only binary hidden units are supported");

    //Configurable properties
    weight learning_rate = 1e-1;
    weight momentum = 0.5;

    etl::fast_vector<etl::fast_vector<weight, NW * NW>, K> w;     //shared weights
    etl::fast_vector<weight, K> b;                           //hidden biases bk
    weight c;                                           //visible single bias c

    etl::fast_vector<weight, NV * NV> v1;                    //visible units

    etl::fast_vector<etl::fast_vector<weight, NH * NH>, K> h1_a;  //Activation probabilities of reconstructed hidden units
    etl::fast_vector<etl::fast_vector<weight, NH * NH>, K> h1_s;  //Sampled values of reconstructed hidden units

    etl::fast_vector<weight, NV * NV> v2_a;                  //Activation probabilities of reconstructed visible units
    etl::fast_vector<weight, NV * NV> v2_s;                  //Sampled values of reconstructed visible units

    etl::fast_vector<etl::fast_vector<weight, NH * NH>, K> h2_a;  //Activation probabilities of reconstructed hidden units
    etl::fast_vector<etl::fast_vector<weight, NH * NH>, K> h2_s;  //Sampled values of reconstructed hidden units

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