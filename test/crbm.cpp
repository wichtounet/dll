//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "catch.hpp"

#include "dll/cpp_utils/data.hpp"

#include "dll/conv_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "crbm/mnist_1", "crbm::simple" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 2e-2);
}

TEST_CASE( "crbm/mnist_2", "crbm::momentum" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
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

TEST_CASE( "crbm/mnist_3", "crbm::decay_l1" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
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

TEST_CASE( "crbm/mnist_4", "crbm::decay_l2" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
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

TEST_CASE( "crbm/mnist_60", "crbm::global_sparsity" ) {
    using rbm_type = dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>,
        dll::sparsity<>
    >::rbm_t;

    REQUIRE(dll::rbm_traits<rbm_type>::sparsity_method() == dll::sparsity_method::GLOBAL_TARGET);

    rbm_type rbm;

    //0.01 (default) is way too low for few hidden units
    rbm.sparsity_target = 0.1;
    rbm.sparsity_cost = 0.9;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm/mnist_61", "crbm::local_sparsity" ) {
    using rbm_type = dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>,
        dll::sparsity<dll::sparsity_method::LOCAL_TARGET>
    >::rbm_t;

    rbm_type rbm;

    //0.01 (default) is way too low for few hidden units
    rbm.sparsity_target = 0.1;
    rbm.sparsity_cost = 0.9;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm/mnist_6", "crbm::gaussian" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::visible<dll::unit_type::GAUSSIAN>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 2e-2);
}

TEST_CASE( "crbm/mnist_7", "crbm::relu" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU>
    >::rbm_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm/mnist_8", "crbm::relu6" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU6>
    >::rbm_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-3);
}

TEST_CASE( "crbm/mnist_9", "crbm::relu1" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
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

TEST_CASE( "crbm/mnist_10", "crbm::pcd" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>,
        dll::momentum,
        dll::trainer<dll::pcd1_trainer_t>
    >::rbm_t rbm;

    rbm.learning_rate *= 0.01;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

TEST_CASE( "crbm/mnist_11", "crbm::bias_mode_simple" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>,
        dll::momentum,
        dll::bias<dll::bias_mode::SIMPLE>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

TEST_CASE( "crbm/mnist_12", "crbm::bias_mode_none" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>,
        dll::momentum,
        dll::bias<dll::bias_mode::NONE>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

TEST_CASE( "crbm/mnist_13", "crbm::multi_channel" ) {
    dll::conv_rbm_desc_square<
        28, 2, 12, 40,
        dll::batch_size<25>,
        dll::momentum
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(200);

    mnist::binarize_dataset(dataset);

    for(auto& image : dataset.training_images){
        image.reserve(image.size() * 2);
        auto end = image.size();
        for(std::size_t i = 0; i < end; ++i){
            image.push_back(image[i]);
        }
    }

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm/mnist_14", "crbm::fast" ) {
    dll::conv_rbm_desc_square<
        28, 2, 12, 40,
        dll::batch_size<25>,
        dll::momentum
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(200);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    for(auto& image : dataset.training_images){
        image.reserve(image.size() * 2);
        auto end = image.size();
        for(std::size_t i = 0; i < end; ++i){
            image.push_back(image[i]);
        }
    }

    auto error = rbm.train(dataset.training_images, 25);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "crbm/mnist_15", "crbm::denoising" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::visible<dll::unit_type::GAUSSIAN>,
        dll::shuffle
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

    auto error = rbm.train_denoising(noisy, dataset.training_images, 100);

    REQUIRE(error < 2e-2);
}

TEST_CASE( "crbm/mnist_16", "crbm::momentum" ) {
    dll::conv_rbm_desc_square<
        28, 1, 12, 40,
        dll::batch_size<25>,
        dll::momentum,
        dll::parallel
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 2e-2);
}
