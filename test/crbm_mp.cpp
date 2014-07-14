#include "catch.hpp"

#include "dll/conv_rbm_mp.hpp"
#include "dll/vector.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "crbm_mp/mnist_1", "crbm::simple" ) {
    dll::conv_mp_layer<
        28, 12, 40, 2,
        dll::batch_size<25>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm_mp/mnist_2", "crbm::momentum" ) {
    dll::conv_mp_layer<
        28, 12, 40, 2,
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

TEST_CASE( "crbm_mp/mnist_3", "crbm::decay_l1" ) {
    dll::conv_mp_layer<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::weight_decay<dll::decay_type::L1_FULL>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm_mp/mnist_4", "crbm::decay_l2" ) {
    dll::conv_mp_layer<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::weight_decay<dll::decay_type::L2_FULL>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm_mp/mnist_5", "crbm::sparsity" ) {
    dll::conv_mp_layer<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::sparsity
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "crbm_mp/mnist_6", "crbm::gaussian" ) {
    dll::conv_mp_layer<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::visible<dll::unit_type::GAUSSIAN>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "crbm_mp/mnist_7", "crbm::relu" ) {
    //TODO This does not work right now

    dll::conv_mp_layer<
        28, 12, 40, 2,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}