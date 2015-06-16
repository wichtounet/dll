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

TEST_CASE( "rbm/mnist_60", "rbm::global_sparsity" ) {
    using rbm_type = dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::sparsity<>
    >::rbm_t;

    rbm_type rbm;

	//Ensure that the default is correct
    REQUIRE(dll::layer_traits<rbm_type>::sparsity_method() == dll::sparsity_method::GLOBAL_TARGET);

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

TEST_CASE( "rbm/mnist_61", "rbm::local_sparsity" ) {
    dll::rbm_desc<
        28 * 28, 100,
       dll::batch_size<25>,
       dll::sparsity<dll::sparsity_method::LOCAL_TARGET>
    >::rbm_t rbm;

    //0.01 (default) is way too low for 100 hidden units
    rbm.sparsity_target = 0.1;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(100);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 1e-2);
}

//TODO Still not very convincing
TEST_CASE( "rbm/mnist_62", "rbm::sparsity_gaussian" ) {
    dll::rbm_desc<
        28 * 28, 200,
       dll::batch_size<25>,
       dll::momentum,
       dll::sparsity<>,
       dll::visible<dll::unit_type::GAUSSIAN>
    >::rbm_t rbm;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(500);

    mnist::normalize_dataset(dataset);

    auto error = rbm.train(dataset.training_images, 200);

    REQUIRE(error < 1e-2);
}
