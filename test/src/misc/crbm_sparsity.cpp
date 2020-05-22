//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/rbm/conv_rbm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("crbm/mnist_60", "crbm::global_sparsity") {
    using rbm_type = dll::conv_rbm_square_desc<
        1, 28, 40, 17,
        dll::batch_size<25>,
        dll::sparsity<>>::layer_t;

    REQUIRE(dll::rbm_layer_traits<rbm_type>::sparsity_method() == dll::sparsity_method::GLOBAL_TARGET);

    rbm_type rbm;

    //0.01 (default) is way too low for few hidden units
    rbm.sparsity_target = 0.1;
    rbm.sparsity_cost   = 0.9;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("crbm/mnist_61", "crbm::local_sparsity") {
    using rbm_type = dll::conv_rbm_square_desc<
        1, 28, 40, 17,
        dll::batch_size<25>,
        dll::sparsity<dll::sparsity_method::LOCAL_TARGET>>::layer_t;

    rbm_type rbm;

    //0.01 (default) is way too low for few hidden units
    rbm.sparsity_target = 0.1;
    rbm.sparsity_cost   = 0.9;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("crbm/mnist_11", "crbm::bias_mode_simple") {
    dll::conv_rbm_square_desc<
        1, 28, 40, 17,
        dll::batch_size<25>,
        dll::momentum,
        dll::bias<dll::bias_mode::SIMPLE>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}

DLL_TEST_CASE("crbm/mnist_12", "crbm::bias_mode_none") {
    dll::conv_rbm_square_desc<
        1, 28, 40, 17,
        dll::batch_size<25>,
        dll::momentum,
        dll::bias<dll::bias_mode::NONE>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 5e-2);
}
