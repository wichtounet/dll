//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/rbm/dyn_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("dyn_rbm/mnist_1", "rbm::simple") {
    dll::dyn_rbm_desc<>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("dyn_rbm/mnist_2", "rbm::momentum") {
    dll::dyn_rbm_desc<dll::momentum>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("dyn_rbm/mnist_3", "rbm::pcd_trainer") {
    dll::dyn_rbm_desc<
        dll::momentum,
        dll::trainer_rbm<dll::pcd1_trainer_t>>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}

DLL_TEST_CASE("dyn_rbm/mnist_4", "rbm::decay_l1") {
    dll::dyn_rbm_desc<
        dll::weight_decay<dll::decay_type::L1>>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("dyn_rbm/mnist_5", "rbm::decay_l2") {
    dll::dyn_rbm_desc<
        dll::weight_decay<dll::decay_type::L2>>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("dyn_rbm/mnist_60", "rbm::global_sparsity") {
    using layer_type = dll::dyn_rbm_desc<
        dll::sparsity<>>::layer_t;

    layer_type rbm(28 * 28, 100);

    REQUIRE(dll::rbm_layer_traits<layer_type>::sparsity_method() == dll::sparsity_method::GLOBAL_TARGET);

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("dyn_rbm/mnist_61", "rbm::local_sparsity") {
    dll::dyn_rbm_desc<
        dll::sparsity<dll::sparsity_method::LOCAL_TARGET>>::layer_t rbm(28 * 28, 100);

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("dyn_rbm/mnist_7", "rbm::gaussian") {
    dll::dyn_rbm_desc<
        dll::visible<dll::unit_type::GAUSSIAN>>::layer_t rbm(28 * 28, 100);

    rbm.learning_rate *= 10;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("dyn_rbm/mnist_8", "rbm::softmax") {
    dll::dyn_rbm_desc<
        dll::hidden<dll::unit_type::SOFTMAX>>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("dyn_rbm/mnist_9", "rbm::relu") {
    dll::dyn_rbm_desc<
        dll::hidden<dll::unit_type::RELU>>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

DLL_TEST_CASE("dyn_rbm/mnist_10", "rbm::relu1") {
    dll::dyn_rbm_desc<
        dll::hidden<dll::unit_type::RELU1>>::layer_t rbm(28 * 28, 100);

    rbm.learning_rate *= 2.0;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

DLL_TEST_CASE("dyn_rbm/mnist_11", "rbm::relu6") {
    dll::dyn_rbm_desc<
        dll::hidden<dll::unit_type::RELU6>>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-1);
}

DLL_TEST_CASE("dyn_rbm/mnist_12", "rbm::init_weights") {
    dll::dyn_rbm_desc<
        dll::init_weights>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-3);
}

//Only here for debugging purposes
DLL_TEST_CASE("dyn_rbm/mnist_15", "rbm::fast") {
    dll::dyn_rbm_desc<>::layer_t rbm(28 * 28, 100);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 5);

    REQUIRE(error < 5e-1);
}
