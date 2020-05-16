//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*
 * This is mostly a compilation test to ensure that RBM is accepting
 * enough input types
 */

#include <vector>
#include <list>
#include <deque>

#include "dll_test.hpp"
#include "template_test.hpp"

#include "dll/rbm/rbm.hpp"
#include "dll/rbm/dyn_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

struct rbm_double {
    using rbm_t = dll::rbm_desc<
            28 * 28, 100,
            dll::weight_type<double>,
            dll::batch_size<25>>::layer_t;

    static void init(rbm_t&){}
};

struct rbm_float {
    using rbm_t = dll::rbm_desc<
            28 * 28, 100,
            dll::weight_type<float>,
            dll::batch_size<25>>::layer_t;

    static void init(rbm_t&){}
};

struct dyn_rbm_float {
    using rbm_t = dll::dyn_rbm_desc<
            dll::weight_type<float>
            , dll::batch_size<25>
            >::layer_t;

    static void init(rbm_t& rbm){
        rbm.init_layer(28 * 28, 100);
    }
};

struct dyn_rbm_double {
    using rbm_t = dll::dyn_rbm_desc<
            dll::weight_type<double>
            , dll::batch_size<25>
            >::layer_t;

    static void init(rbm_t& rbm){
        rbm.init_layer(28 * 28, 100);
    }
};

} // end of anonymous namespace

#define TYPES_TEST_PREFIX "rbm"
#define FLOAT_TYPES_TEST_T1 rbm_float
#define FLOAT_TYPES_TEST_T2 dyn_rbm_float
#define DOUBLE_TYPES_TEST_T1 rbm_double
#define DOUBLE_TYPES_TEST_T2 dyn_rbm_double

#include "types_test.inl"
