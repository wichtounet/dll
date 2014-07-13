#include "catch.hpp"

#include "dll/rbm.hpp"
#include "dll/layer.hpp"
#include "dll/vector.hpp"
#include "dll/generic_trainer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "rbm/mnist_1", "rbm::simple" ) {
    dll::rbm<dll::layer<
            28 * 28, 100,
            dll::batch_size<25>
            >> rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_2", "rbm::momentum" ) {
    dll::rbm<dll::layer<
            28 * 28, 100,
            dll::batch_size<25>,
            dll::momentum
            >> rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_3", "rbm::pcd_trainer" ) {
    dll::rbm<dll::layer<
            28 * 28, 100,
            dll::batch_size<25>,
            dll::momentum,
            dll::trainer<dll::pcd1_trainer_t>
            >> rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}