//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "dll_test.hpp"

#include "cpp_utils/data.hpp"

#include "dll/rbm/conv_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/crbm/mnist/1", "[crbm][unit]") {
    dll::conv_rbm_desc_square<
        1, 28, 20, 17,
        dll::batch_size<10>,
        dll::weight_decay<dll::decay_type::L2_FULL>,
        dll::momentum>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);
    REQUIRE(error < 5e-2);

    rbm.v1 = dataset.training_images[1];

    rbm.template activate_hidden<true, false>(rbm.h1_a, rbm.h1_a, rbm.v1, rbm.v1);

    auto energy = rbm.energy(dataset.training_images[1], rbm.h1_a);
    REQUIRE(energy < 0.0);

    auto free_energy = rbm.free_energy();
    REQUIRE(free_energy < 0.0);
}

TEST_CASE("unit/crbm/mnist/2", "[crbm][parallel][unit]") {
    dll::conv_rbm_desc_square<
        1, 28, 20, 5,
        dll::batch_size<25>,
        dll::momentum,
        dll::parallel_mode,
        dll::weight_decay<dll::decay_type::L2>,
        dll::visible<dll::unit_type::GAUSSIAN>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 0.1);
}

TEST_CASE("unit/crbm/mnist/4", "[crbm][unit]") {
    dll::conv_rbm_desc_square<
        1, 28, 20, 17,
        dll::batch_size<25>,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::visible<dll::unit_type::GAUSSIAN>,
        dll::shuffle>::layer_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(200);

    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto noisy = dataset.training_images;

    std::default_random_engine rand_engine(56);
    std::normal_distribution<float> normal_distribution(0.0, 0.1);
    auto noise = std::bind(normal_distribution, rand_engine);

    for (auto& image : noisy) {
        for (auto& noisy_x : image) {
            noisy_x += noise();
        }
    }

    cpp::normalize_each(noisy);

    auto error = rbm.train_denoising(noisy, dataset.training_images, 50);
    REQUIRE(error < 0.1);
}

TEST_CASE("unit/crbm/mnist/5", "[crbm][unit]") {
    dll::conv_rbm_desc_square<
        1, 28, 20, 9,
        dll::batch_size<20>,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::shuffle,
        dll::hidden<dll::unit_type::RELU>>::layer_t rbm;

    rbm.learning_rate *= 5;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 25);
    REQUIRE(error < 5e-2);
}

TEST_CASE("unit/crbm/mnist/6", "[crbm][unit]") {
    using layer_type = dll::conv_rbm_desc_square<
        1, 28, 20, 17,
        dll::batch_size<10>,
        dll::sparsity<>>::layer_t;

    REQUIRE(dll::rbm_layer_traits<layer_type>::sparsity_method() == dll::sparsity_method::GLOBAL_TARGET);

    layer_type rbm;

    //0.01 (default) is way too low for few hidden units
    rbm.sparsity_target = 0.1;
    rbm.sparsity_cost   = 0.9;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    // Note: Not an excellent performance is expected due to the
    // large kernel and high sparsity goal

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 0.1);
}

TEST_CASE("unit/crbm/mnist/7", "[crbm][unit]") {
    using layer_type = dll::conv_rbm_desc_square<
        1, 28, 20, 17,
        dll::batch_size<5>,
        dll::sparsity<dll::sparsity_method::LOCAL_TARGET>>::layer_t;

    layer_type rbm;

    //0.01 (default) is way too low for few hidden units
    rbm.sparsity_target = 0.1;
    rbm.sparsity_cost   = 0.9;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 7e-2);
}
