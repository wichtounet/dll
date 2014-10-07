//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "dll/conv_rbm_mp.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "crbm_mp/mnist_1", "crbm::simple" ) {
    dll::conv_rbm_mp_desc<
        28, 12, 40, 2,
        dll::batch_size<25>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm_mp/mnist_2", "crbm::momentum" ) {
    dll::conv_rbm_mp_desc<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::momentum
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm_mp/mnist_3", "crbm::decay_l1" ) {
    dll::conv_rbm_mp_desc<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::weight_decay<dll::decay_type::L1_FULL>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm_mp/mnist_4", "crbm::decay_l2" ) {
    dll::conv_rbm_mp_desc<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::weight_decay<dll::decay_type::L2_FULL>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm_mp/mnist_5", "crbm::sparsity" ) {
    dll::conv_rbm_mp_desc<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::sparsity<>
    >::rbm_t rbm;

    //0.01 (default) is way too low for few hidden units
    rbm.sparsity_target = 0.1;
    rbm.sparsity_cost = 0.9;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "crbm_mp/mnist_6", "crbm::gaussian" ) {
    dll::conv_rbm_mp_desc<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::visible<dll::unit_type::GAUSSIAN>
    >::rbm_t rbm;

    rbm.learning_rate *= 1.5;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "crbm_mp/mnist_7", "crbm::relu" ) {
    //TODO This does not work right now

    dll::conv_rbm_mp_desc<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}

//TODO Test other ReLU versions

TEST_CASE( "crbm_mp/mnist_10", "crbm::pcd_trainer" ) {
    //TODO This test does not work right now

    dll::conv_rbm_mp_desc<
        28, 12, 40, 2,
        dll::batch_size<10>,
        dll::momentum,
        dll::trainer<dll::pcd1_trainer_t>
    >::rbm_t rbm;

    rbm.learning_rate /= 100.0;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(200);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm_mp/mnist_11", "crbm::bias_mode_none" ) {
    dll::conv_rbm_mp_desc<
        28, 12, 40, 2,
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

TEST_CASE( "crbm_mp/mnist_12", "crbm::bias_mode_simple" ) {
    dll::conv_rbm_mp_desc<
        28, 12, 40, 2,
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