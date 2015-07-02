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

TEST_CASE( "crbm_mp/mnist_5", "crbm::sparsity" ) {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 12, 2,
        dll::batch_size<25>,
        dll::sparsity<>
    >::rbm_t rbm;

    //0.01 (default) is way too low for few hidden units
    rbm.sparsity_target = 0.1;
    rbm.sparsity_cost = 0.9;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "crbm_mp/mnist_110", "crbm::bias_mode_none" ) {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 12, 2,
        dll::batch_size<10>,
        dll::momentum,
        dll::sparsity<dll::sparsity_method::LEE>,
        dll::bias<dll::bias_mode::NONE>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(200);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm_mp/mnist_111", "crbm::bias_mode_simple" ) {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 12, 2,
        dll::batch_size<10>,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::sparsity<dll::sparsity_method::LEE>,
        dll::bias<dll::bias_mode::SIMPLE>
    >::rbm_t rbm;

    rbm.l2_weight_cost = 0.01;
    rbm.learning_rate = 0.01;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(200);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm_mp/mnist_12", "crbm::lee" ) {
    //This test is not meant to be stable, just use it to experiment with
    //sparsity / gaussian

    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 12, 2,
        dll::batch_size<5>,
        dll::momentum,
        dll::visible<dll::unit_type::GAUSSIAN>,
        dll::weight_decay<dll::decay_type::L2>,
        dll::sparsity<dll::sparsity_method::LEE>,
        dll::bias<dll::bias_mode::SIMPLE>
    >::rbm_t rbm;

    rbm.pbias = 0.01;
    rbm.pbias_lambda = 0.1;
    //rbm.learning_rate = 0.01;
    rbm.learning_rate *= 12;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);
    //mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}
