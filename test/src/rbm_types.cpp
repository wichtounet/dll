//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*
 * This is mostly a compilation test to ensure that RBM is accepting
 * enough input types
 */

#include <numeric>
#include <vector>
#include <list>
#include <deque>

#include "catch.hpp"

#include "cpp_utils/data.hpp"

#include "dll/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace {

using rbm_float = dll::rbm_desc<
        28 * 28, 100,
        dll::weight_type<float>,
        dll::batch_size<25>>::layer_t;

} // end of anonymous namespace

// fast_rbm<float> <- std::vector<float>
TEST_CASE("rbm/types/1", "[types]") {
    rbm_float rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<float>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.1);
    REQUIRE(rbm.reconstruction_error(sample) < 0.1);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    auto b = rbm.features(sample);

    REQUIRE(rbm.free_energy(sample) < 0.0);
    REQUIRE(rbm.energy(sample, a) > 0.0);
    REQUIRE(rbm.energy(sample, b) > 0.0);
}

// fast_rbm<float> <- std::list<float>
TEST_CASE("rbm/types/2", "[types]") {
    rbm_float rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<float>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    // mnist reader does not support std::list (with reason)
    std::vector<std::list<float>> training_images;
    training_images.reserve(dataset.training_images.size());
    for(auto& image : dataset.training_images){
        training_images.emplace_back(image.begin(), image.end());
    }

    REQUIRE(rbm.train(training_images, 20) < 0.1);
    REQUIRE(rbm.reconstruction_error(sample) < 0.1);
    REQUIRE(rbm.train_denoising(training_images, training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    auto b = rbm.features(sample);

    REQUIRE(rbm.free_energy(sample) < 0.0);
    REQUIRE(rbm.energy(sample, a) > 0.0);
    REQUIRE(rbm.energy(sample, b) > 0.0);
}

// fast_rbm<float> <- std::deque<float>
TEST_CASE("rbm/types/3", "[types]") {
    rbm_float rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, std::deque<float>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.1);
    REQUIRE(rbm.reconstruction_error(sample) < 0.1);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    auto b = rbm.features(sample);

    REQUIRE(rbm.free_energy(sample) < 0.0);
    REQUIRE(rbm.energy(sample, a) > 0.0);
    REQUIRE(rbm.energy(sample, b) > 0.0);
}

// fast_rbm<float> <- std::vector<double>
TEST_CASE("rbm/types/4", "[types]") {
    rbm_float rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<double>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.1);
    REQUIRE(rbm.reconstruction_error(sample) < 0.1);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    auto b = rbm.features(sample);

    REQUIRE(rbm.free_energy(sample) < 0.0);
    REQUIRE(rbm.energy(sample, a) > 0.0);
    REQUIRE(rbm.energy(sample, b) > 0.0);
}

// fast_rbm<float> <- std::list<double>
TEST_CASE("rbm/types/5", "[types]") {
    rbm_float rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<double>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    // mnist reader does not support std::list (with reason)
    std::vector<std::list<double>> training_images;
    training_images.reserve(dataset.training_images.size());
    for(auto& image : dataset.training_images){
        training_images.emplace_back(image.begin(), image.end());
    }

    REQUIRE(rbm.train(training_images, 20) < 0.1);
    REQUIRE(rbm.reconstruction_error(sample) < 0.1);
    REQUIRE(rbm.train_denoising(training_images, training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    auto b = rbm.features(sample);

    REQUIRE(rbm.free_energy(sample) < 0.0);
    REQUIRE(rbm.energy(sample, a) > 0.0);
    REQUIRE(rbm.energy(sample, b) > 0.0);
}

// fast_rbm<float> <- std::deque<double>
TEST_CASE("rbm/types/6", "[types]") {
    rbm_float rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, std::deque<double>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.1);
    REQUIRE(rbm.reconstruction_error(sample) < 0.1);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    auto b = rbm.features(sample);

    REQUIRE(rbm.free_energy(sample) < 0.0);
    REQUIRE(rbm.energy(sample, a) > 0.0);
    REQUIRE(rbm.energy(sample, b) > 0.0);
}

// fast_rbm<float> <- etl::dyn_matrix<float, 1>
TEST_CASE("rbm/types/7", "[types]") {
    rbm_float rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.1);
    REQUIRE(rbm.reconstruction_error(sample) < 0.1);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    auto b = rbm.features(sample);

    REQUIRE(rbm.free_energy(sample) < 0.0);
    REQUIRE(rbm.energy(sample, a) > 0.0);
    REQUIRE(rbm.energy(sample, b) > 0.0);
}

// fast_rbm<float> <- etl::fast_dyn_matrix<float, 1>
TEST_CASE("rbm/types/8", "[types]") {
    rbm_float rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.1);
    REQUIRE(rbm.reconstruction_error(sample) < 0.1);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    auto b = rbm.features(sample);

    REQUIRE(rbm.free_energy(sample) < 0.0);
    REQUIRE(rbm.energy(sample, a) > 0.0);
    REQUIRE(rbm.energy(sample, b) > 0.0);
}

// fast_rbm<float> <- etl::dyn_matrix<double, 1>
TEST_CASE("rbm/types/9", "[types]") {
    rbm_float rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<double, 1>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.1);
    REQUIRE(rbm.reconstruction_error(sample) < 0.1);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    auto b = rbm.features(sample);

    REQUIRE(rbm.free_energy(sample) < 0.0);
    REQUIRE(rbm.energy(sample, a) > 0.0);
    REQUIRE(rbm.energy(sample, b) > 0.0);
}

// fast_rbm<float> <- etl::fast_dyn_matrix<double, 1>
TEST_CASE("rbm/types/10", "[types]") {
    rbm_float rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 28 * 28>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.1);
    REQUIRE(rbm.reconstruction_error(sample) < 0.1);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    auto b = rbm.features(sample);

    REQUIRE(rbm.free_energy(sample) < 0.0);
    REQUIRE(rbm.energy(sample, a) > 0.0);
    REQUIRE(rbm.energy(sample, b) > 0.0);
}
