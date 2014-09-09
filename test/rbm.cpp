//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "dll/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "rbm/mnist_1", "rbm::simple" ) {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_2", "rbm::momentum" ) {
    dll::rbm_desc<
        28 * 28, 100,
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

TEST_CASE( "rbm/mnist_3", "rbm::pcd_trainer" ) {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::momentum,
        dll::trainer<dll::pcd1_trainer_t>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "rbm/mnist_4", "rbm::decay_l1" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::weight_decay<dll::decay_type::L1>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_5", "rbm::decay_l2" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::weight_decay<dll::decay_type::L2>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_6", "rbm::sparsity" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::sparsity
    >::rbm_t rbm;

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_7", "rbm::gaussian" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::visible<dll::unit_type::GAUSSIAN>
    >::rbm_t rbm;

    rbm.learning_rate *= 10;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_8", "rbm::softmax" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::hidden<dll::unit_type::SOFTMAX>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_9", "rbm::relu" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::hidden<dll::unit_type::RELU>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "rbm/mnist_10", "rbm::relu1" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::hidden<dll::unit_type::RELU1>
    >::rbm_t rbm;

    rbm.learning_rate *= 2.0;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "rbm/mnist_11", "rbm::relu6" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::hidden<dll::unit_type::RELU6>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "rbm/mnist_12", "rbm::init_weights" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::init_weights
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-3);
}

TEST_CASE( "rbm/mnist_13", "rbm::exp" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::hidden<dll::unit_type::EXP>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    //This test is kind of fake since exp unit are not really made for
    //reconstruction. It is here to ensure that exp units are working.
    //exponential units are not even made for training

    REQUIRE(std::isnan(error));
}