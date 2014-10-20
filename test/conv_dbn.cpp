//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "catch.hpp"

#include "dll/conv_rbm.hpp"
#include "dll/conv_dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "conv_dbn/mnist_1", "conv_dbn::simple" ) {
    typedef dll::conv_dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc<28, 1, 12, 40, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc<12, 40, 10, 20, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc<10, 20, 6, 50, dll::momentum, dll::batch_size<25>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);
    dataset.training_labels.resize(100);

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 5);
}
