//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#define DLL_PARALLEL

#include "dll/rbm.hpp"
#include "dll/dyn_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "rbm/mnist_1", "rbm::simple" ) {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

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

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

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

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "rbm/mnist_40", "rbm::decay_l1" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::weight_decay<dll::decay_type::L1>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_41", "rbm::decay_l2" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::weight_decay<dll::decay_type::L2>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_42", "rbm::decay_l1l2" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::weight_decay<dll::decay_type::L1L2>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_43", "rbm::decay_l1l2_full" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::weight_decay<dll::decay_type::L1L2_FULL>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_60", "rbm::global_sparsity" ) {
    using rbm_type = dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::sparsity<>
    >::rbm_t;

    rbm_type rbm;

	//Ensure that the default is correct
    REQUIRE(dll::rbm_traits<rbm_type>::sparsity_method() == dll::sparsity_method::GLOBAL_TARGET);

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_61", "rbm::local_sparsity" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::sparsity<dll::sparsity_method::LOCAL_TARGET>
    >::rbm_t rbm;

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_7", "rbm::gaussian" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::momentum,
       dll::visible<dll::unit_type::GAUSSIAN>
    >::rbm_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);

    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "rbm/mnist_8", "rbm::softmax" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::hidden<dll::unit_type::SOFTMAX>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

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

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_10", "rbm::relu1" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::hidden<dll::unit_type::RELU1>
    >::rbm_t rbm;

    rbm.learning_rate *= 2.0;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

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

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

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

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_13", "rbm::exp" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::hidden<dll::unit_type::EXP>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 10);

    //This test is kind of fake since exp unit are not really made for
    //reconstruction. It is here to ensure that exp units are working.
    //exponential units are not even made for training

    REQUIRE(std::isnan(error));
}

//TODO Still not very convincing
TEST_CASE( "rbm/mnist_14", "rbm::sparsity_gaussian" ) {
    dll::rbm_desc<
        28 * 28, 200,
       dll::batch_size<25>,
       dll::momentum,
       dll::sparsity<>,
       dll::visible<dll::unit_type::GAUSSIAN>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(500);

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-2);
}

//TODO Still not very convincing
TEST_CASE( "rbm/mnist_15", "rbm::pcd_gaussian" ) {
    dll::rbm_desc<
        28 * 28, 144,
       dll::batch_size<25>,
       dll::momentum,
       dll::trainer<dll::pcd1_trainer_t>,
       dll::visible<dll::unit_type::GAUSSIAN>
    >::rbm_t rbm;

    rbm.learning_rate /= 20.0;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

TEST_CASE( "rbm/mnist_16", "rbm::iterators" ) {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<23>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto it = dataset.training_images.begin();
    auto end = dataset.training_images.end();

    auto error = rbm.train(it, end, 100);

    REQUIRE(error < 1e-2);
}

//Only here for benchmarking purpose
TEST_CASE( "rbm/mnist_17", "rbm::slow" ) {
    dll::rbm_desc<
        28 * 28, 400,
        dll::batch_size<48>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 10);

    REQUIRE(error < 5e-2);
}

//Only here for debugging purposes
TEST_CASE( "rbm/mnist_18", "rbm::fast" ) {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<50>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(25);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 5);

    REQUIRE(error < 5e-1);
}

TEST_CASE( "rbm/mnist_19", "rbm::simple_double" ) {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::weight_type<double>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 3e-2);
}

TEST_CASE( "rbm/mnist_20", "rbm::simple_float" ) {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::weight_type<float>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 3e-2);
}

TEST_CASE( "rbm/mnist_21", "rbm::shuffle" ) {
    dll::rbm_desc<
        28 * 28, 400,
        dll::batch_size<48>,
        dll::shuffle
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(1000);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 10);

    REQUIRE(error < 5e-2);
}
