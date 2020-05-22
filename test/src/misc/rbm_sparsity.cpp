//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/rbm/rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("rbm/mnist_60", "rbm::global_sparsity") {
    using rbm_type = dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::sparsity<>>::layer_t;

    rbm_type rbm;

    //Ensure that the default is correct
    REQUIRE(dll::rbm_layer_traits<rbm_type>::sparsity_method() == dll::sparsity_method::GLOBAL_TARGET);

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("rbm/mnist_61", "rbm::local_sparsity") {
    dll::rbm_desc<
        28 * 28, 100,
        dll::batch_size<25>,
        dll::sparsity<dll::sparsity_method::LOCAL_TARGET>>::layer_t rbm;

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("rbm/mnist_62", "rbm::sparsity_gaussian") {
    dll::rbm_desc<
        28 * 28, 300,
        dll::batch_size<10>,
        dll::momentum,
        dll::sparsity<dll::sparsity_method::LOCAL_TARGET>,
        dll::visible<dll::unit_type::GAUSSIAN>>::layer_t rbm;

    rbm.learning_rate *= 2;
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(500);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 0.25);
}
