//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "cpp_utils/data.hpp"

#include "dll/rbm/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("rbm/mnist_1", "rbm::simple") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::verbose>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 3e-2);

    auto rec_error = rbm.reconstruction_error(dataset.training_images[1]);

    REQUIRE(rec_error < 3e-2);
}

DLL_TEST_CASE("rbm/mnist_2", "rbm::momentum") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::momentum>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("rbm/mnist_40", "rbm::decay_l1") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::weight_decay<dll::decay_type::L1>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("rbm/mnist_41", "rbm::decay_l2") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::weight_decay<dll::decay_type::L2>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("rbm/mnist_42", "rbm::decay_l1l2") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::weight_decay<dll::decay_type::L1L2>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("rbm/mnist_43", "rbm::decay_l1l2_full") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::weight_decay<dll::decay_type::L1L2_FULL>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("rbm/mnist_7", "rbm::gaussian") {
    dll::rbm_desc<
        28 * 28, 333,
        dll::batch_size<20>,
        dll::weight_decay<>,
        dll::momentum,
        dll::visible<dll::unit_type::GAUSSIAN>>::layer_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

DLL_TEST_CASE("rbm/mnist_8", "rbm::softmax") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::hidden<dll::unit_type::SOFTMAX>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("rbm/mnist_12", "rbm::init_weights") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::init_weights>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("rbm/mnist_16", "rbm::iterators") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto it  = dataset.training_images.begin();
    auto end = dataset.training_images.end();

    auto error = rbm.train(it, end, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("rbm/mnist_19", "rbm::simple_double") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::weight_type<double>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<double>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 3e-2);
}

DLL_TEST_CASE("rbm/mnist_20", "rbm::simple_float") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::weight_type<float>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 3e-2);
}

DLL_TEST_CASE("rbm/mnist_21", "rbm::shuffle") {
    dll::rbm_desc<
        28 * 28, 400,
        dll::batch_size<48>,
        dll::shuffle>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 10);

    REQUIRE(error < 5e-2);
}

DLL_TEST_CASE("rbm/mnist_22", "rbm::denoising") {
    dll::rbm_desc<
        28 * 28, 200,
        dll::batch_size<25>,
        dll::momentum,
        dll::weight_decay<>,
        dll::visible<dll::unit_type::GAUSSIAN>,
        dll::shuffle,
        dll::weight_type<float>>::layer_t rbm;

    rbm.learning_rate *= 2;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(200);
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

    auto error = rbm.train_denoising(noisy, dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}
