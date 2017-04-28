//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "dll_test.hpp"

#include "cpp_utils/data.hpp"

#include "dll/rbm/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/rbm/mnist/1", "[rbm][momentum][unit]") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::momentum,
        dll::shuffle_cond<false>,
        dll::nop>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);

    REQUIRE(error < 1e-2);

    auto rec_error = rbm.reconstruction_error(dataset.training_images[4]);

    REQUIRE(rec_error < 1e-2);
}

TEST_CASE("unit/rbm/mnist/2", "[rbm][momentum][parallel][unit]") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::momentum,
        dll::parallel_mode>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);

    REQUIRE(error < 1e-2);
}

TEST_CASE("unit/rbm/mnist/3", "[rbm][momemtum][gaussian][unit]") {
    dll::rbm_desc<
        28 * 28, 150,
        dll::batch_size<25>,
        dll::momentum,
        dll::visible<dll::unit_type::GAUSSIAN>>::layer_t rbm;

    rbm.learning_rate *= 20;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 1e-1);
}

TEST_CASE("unit/rbm/mnist/4", "[rbm][shuffle][unit]") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::momentum,
        dll::shuffle_cond<true>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 50);
    REQUIRE(error < 5e-2);
}

TEST_CASE("unit/rbm/mnist/5", "[rbm][denoising][unit]") {
    dll::rbm_desc<
        28 * 28, 200,
        dll::batch_size<25>,
        dll::momentum,
        dll::weight_decay<>,
        dll::visible<dll::unit_type::GAUSSIAN>,
        dll::shuffle,
        dll::weight_type<float>>::layer_t rbm;

    rbm.learning_rate *= 5;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
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
    REQUIRE(error < 1e-1);
}

TEST_CASE("unit/rbm/mnist/6", "[rbm][pcd][unit]") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<5>,
        dll::momentum,
        dll::trainer_rbm<dll::pcd1_trainer_t>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    if (std::isfinite(error)) {
        REQUIRE(error < 15e-2);
    }
}

TEST_CASE("unit/rbm/mnist/7", "[rbm][relu][unit]") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::RELU>>::layer_t rbm;

    rbm.learning_rate *= 10;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE("unit/rbm/mnist/8", "[rbm][sparse][unit]") {
    using rbm_type = dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::sparsity<>>::layer_t;

    rbm_type rbm;

    //Ensure that the default is correct
    REQUIRE(dll::rbm_layer_traits<rbm_type>::sparsity_method() == dll::sparsity_method::GLOBAL_TARGET);

    rbm.learning_rate *= 2;

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

TEST_CASE("unit/rbm/mnist/9", "[rbm][sparse][unit]") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::sparsity<dll::sparsity_method::LOCAL_TARGET>>::layer_t rbm;

    rbm.learning_rate *= 2;

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

template <typename RBM, bool Denoising>
using pcd2_trainer_t = dll::persistent_cd_trainer<2, RBM, Denoising>;

TEST_CASE("unit/rbm/mnist/10", "[rbm][pcd][unit]") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<5>,
        dll::momentum,
        dll::trainer_rbm<pcd2_trainer_t>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    if (std::isfinite(error)) {
        REQUIRE(error < 15e-2);
    }
}
