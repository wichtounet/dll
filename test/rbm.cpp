#include "catch.hpp"

#include "dll/rbm.hpp"
#include "dll/vector.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "rbm/mnist_1", "rbm::simple" ) {
    dll::layer<
        28 * 28, 100,
        dll::batch_size<25>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_2", "rbm::momentum" ) {
    dll::layer<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::momentum
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_3", "rbm::pcd_trainer" ) {
    dll::layer<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::momentum,
        dll::trainer<dll::pcd1_trainer_t>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "rbm/mnist_4", "rbm::decay_l1" ) {
    dll::layer<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::weight_decay<dll::decay_type::L1>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_5", "rbm::decay_l2" ) {
    dll::layer<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::weight_decay<dll::decay_type::L2>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_6", "rbm::sparsity" ) {
    dll::layer<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::sparsity
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_7", "rbm::gaussian" ) {
    dll::layer<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::visible<dll::unit_type::GAUSSIAN>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 7e-2);
}

TEST_CASE( "rbm/mnist_8", "rbm::softmax" ) {
    dll::layer<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::hidden<dll::unit_type::SOFTMAX>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    //This test is kind of fake since softmax unit are not really made for
    //reconstruction. It is here to ensure that softmax units are working.

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_9", "rbm::nrlu" ) {
    dll::layer<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::hidden<dll::unit_type::RELU>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}