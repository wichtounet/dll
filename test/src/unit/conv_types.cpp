//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*
 * This is mostly a compilation test to ensure that DBN is accepting
 * enough input types
 */

#include <vector>
#include <list>
#include <deque>

#include "dll_test.hpp"
#include "template_test.hpp"

#include "dll/neural/conv/conv_layer.hpp"
#include "dll/neural/dense/dense_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

struct dbn_double {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::conv_layer_desc<1, 28, 28, 5, 11, 11, dll::weight_type<double>>::layer_t,
                dll::dense_layer_desc<5 * 18 * 18, 10, dll::weight_type<double>>::layer_t
            >
            , dll::trainer<dll::sgd_trainer>
            , dll::batch_size<10>
            , dll::updater<dll::updater_type::MOMENTUM>
        >::dbn_t;

    static void init(dbn_t& net){
        net.learning_rate = 0.05;
        net.initial_momentum = 0.9;
    }
};

struct dbn_float {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::conv_layer_desc<1, 28, 28, 5, 11, 11, dll::weight_type<float>>::layer_t,
                dll::dense_layer_desc<5 * 18 * 18, 10, dll::weight_type<float>>::layer_t
            >
            , dll::trainer<dll::sgd_trainer>
            , dll::batch_size<10>
            , dll::updater<dll::updater_type::MOMENTUM>
        >::dbn_t;

    static void init(dbn_t& net){
        net.learning_rate = 0.05;
        net.initial_momentum = 0.9;
    }
};

struct dyn_dbn_float {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_conv_layer_desc<dll::weight_type<float>>::layer_t,
                dll::dyn_dense_layer_desc<dll::weight_type<float>>::layer_t
            >
            , dll::trainer<dll::sgd_trainer>
            , dll::batch_size<10>
            , dll::updater<dll::updater_type::MOMENTUM>
        >::dbn_t;

    static void init(dbn_t& net){
        net.initial_momentum = 0.9;
        net.learning_rate = 0.05;
        net.template init_layer<0>(1, 28, 28, 5, 11, 11);
        net.template init_layer<1>(5 * 18 * 18, 10);
    }
};

struct dyn_dbn_double {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_conv_layer_desc<dll::weight_type<double>>::layer_t,
                dll::dyn_dense_layer_desc<dll::weight_type<double>>::layer_t
            >
            , dll::trainer<dll::sgd_trainer>
            , dll::batch_size<10>
            , dll::updater<dll::updater_type::MOMENTUM>
        >::dbn_t;

    static void init(dbn_t& net){
        net.initial_momentum = 0.9;
        net.learning_rate = 0.05;
        net.template init_layer<0>(1, 28, 28, 5, 11, 11);
        net.template init_layer<1>(5 * 18 * 18, 10);
    }
};

} // end of anonymous namespace

#define TYPES_TEST_PREFIX "conv_dbn"
#define FLOAT_TYPES_TEST_T1 dbn_float
#define FLOAT_TYPES_TEST_T2 dyn_dbn_float
#define DOUBLE_TYPES_TEST_T1 dbn_double
#define DOUBLE_TYPES_TEST_T2 dyn_dbn_double
#define TYPES_TEST_NO_PRE

#include "dbn_types_test.inl"
