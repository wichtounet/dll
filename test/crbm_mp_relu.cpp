//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "catch.hpp"

#include "dll/conv_rbm_mp.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "crbm_mp/mnist_7", "crbm::relu" ) {
    dll::conv_rbm_mp_desc_square<
        28, 1, 12, 40, 2,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU>
    >::rbm_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

TEST_CASE( "crbm_mp/mnist_8", "crbm::relu1" ) {
    dll::conv_rbm_mp_desc_square<
        28, 1, 12, 40, 2,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU1>
    >::rbm_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

TEST_CASE( "crbm_mp/mnist_9", "crbm::relu6" ) {
    dll::conv_rbm_mp_desc_square<
        28, 1, 12, 40, 2,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU6>
    >::rbm_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}
