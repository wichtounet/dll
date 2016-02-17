//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "catch.hpp"

#include "cpp_utils/data.hpp"

#include "dll/conv_rbm.hpp"
#include "dll/mp_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("perf/kws", "[perf][crbm][mp][cdbn]") {
    using cdbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::conv_rbm_desc<
                    1, 28, 28, 8, 20, 20, dll::weight_type<float>, dll::batch_size<64>, dll::momentum,
                    dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>, dll::shuffle_cond<true>, dll::dbn_only>::layer_t,
                dll::mp_layer_3d_desc<8, 20, 20, 1, 2, 2, dll::weight_type<float>>::layer_t,
                dll::conv_rbm_desc<
                    8, 10, 10, 8, 8, 8, dll::weight_type<float>, dll::batch_size<64>, dll::momentum,
                    dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>, dll::shuffle_cond<true>, dll::dbn_only>::layer_t,
                dll::mp_layer_3d_desc<8, 8, 8, 1, 2, 2, dll::weight_type<float>>::layer_t>
            >::dbn_t;

    auto cdbn = std::make_unique<cdbn_t>();

    cdbn->display();

    auto dataset = mnist::read_dataset<std::vector, std::vector, float>(8192);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    cdbn->pretrain(dataset.training_images, 5);
}
