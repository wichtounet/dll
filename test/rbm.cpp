//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "catch.hpp"

#include "cpp_utils/data.hpp"

#include "dll/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "rbm/mnist_1", "rbm::simple" ) {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::verbose
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

TEST_CASE( "rbm/mnist_16", "rbm::iterators" ) {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto it = dataset.training_images.begin();
    auto end = dataset.training_images.end();

    auto error = rbm.train(it, end, 100);

    REQUIRE(error < 1e-2);
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

TEST_CASE( "rbm/mnist_22", "rbm::denoising" ) {
    dll::rbm_desc<
        28 * 28, 200,
       dll::batch_size<25>,
       dll::momentum,
       dll::weight_decay<>,
       dll::visible<dll::unit_type::GAUSSIAN>,
       dll::shuffle,
       dll::weight_type<float>
    >::rbm_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);

    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto noisy = dataset.training_images;

    std::default_random_engine rand_engine(56);
    std::normal_distribution<double> normal_distribution(0.0, 0.1);
    auto noise = std::bind(normal_distribution, rand_engine);

    for(auto& image : noisy){
        for(auto& noisy_x : image){
            noisy_x += noise();
        }
    }

    cpp::normalize_each(noisy);

    auto error = rbm.train_denoising(noisy, dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

TEST_CASE( "rbm/mnist_23", "rbm::parallel" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::momentum,
       dll::parallel
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-3);
}
