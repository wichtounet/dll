//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "dll_test.hpp"

#include "dll/rbm/conv_rbm_mp.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("crbm_mp/mnist_5", "crbm::sparsity") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 17, 2,
        dll::batch_size<25>,
        dll::sparsity<>>::layer_t rbm;

    //0.01 (default) is way too low for few hidden units
    rbm.sparsity_target = 0.1;
    rbm.sparsity_cost   = 0.9;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-1);
}

DLL_TEST_CASE("crbm_mp/mnist_110", "crbm::bias_mode_none") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 17, 2,
        dll::batch_size<10>,
        dll::momentum,
        dll::sparsity<dll::sparsity_method::LEE>,
        dll::bias<dll::bias_mode::NONE>>::layer_t rbm;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("crbm_mp/mnist_111", "crbm::bias_mode_simple") {
    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 17, 2,
        dll::batch_size<10>,
        dll::momentum,
        dll::weight_decay<dll::decay_type::L2>,
        dll::sparsity<dll::sparsity_method::LEE>,
        dll::bias<dll::bias_mode::SIMPLE>>::layer_t rbm;

    rbm.l2_weight_cost = 0.01;
    rbm.learning_rate  = 0.01;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

DLL_TEST_CASE("crbm_mp/mnist_12", "crbm::lee") {
    //This test is not meant to be stable, just use it to experiment with
    //sparsity / gaussian

    dll::conv_rbm_mp_desc_square<
        1, 28, 40, 17, 2,
        dll::batch_size<5>,
        dll::momentum,
        dll::visible<dll::unit_type::GAUSSIAN>,
        dll::weight_decay<dll::decay_type::L2>,
        dll::sparsity<dll::sparsity_method::LEE>,
        dll::bias<dll::bias_mode::SIMPLE>>::layer_t rbm;

    rbm.pbias        = 0.01;
    rbm.pbias_lambda = 0.1;
    rbm.learning_rate *= 12;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(200);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}
