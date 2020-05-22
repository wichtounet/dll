//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#define DLL_SVM_SUPPORT

#include "dll/rbm/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("conv_dbn_mp/mnist_slow", "[cdbn][slow][benchmark]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_rbm_mp_desc_square<1, 28, 40, 13, 2, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::conv_rbm_mp_desc_square<40, 8, 40, 5, 2, dll::momentum, dll::batch_size<25>>::layer_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(250);

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
}
