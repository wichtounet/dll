//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <numeric>

#include "dll_test.hpp"

#include "cpp_utils/data.hpp"

#include "dll/rbm/conv_rbm.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("perf/kws_square", "[perf][crbm][mp][cdbn]") {
    using cdbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::conv_rbm_desc<
                    1, 28, 28, 8, 9, 9, dll::weight_type<float>, dll::batch_size<64>, dll::momentum,
                    dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>, dll::shuffle_cond<true>, dll::dbn_only>::layer_t,
                dll::mp_3d_layer_desc<8, 20, 20, 1, 2, 2, dll::weight_type<float>>::layer_t,
                dll::conv_rbm_desc<
                    8, 10, 10, 8, 3, 3, dll::weight_type<float>, dll::batch_size<64>, dll::momentum,
                    dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>, dll::shuffle_cond<true>, dll::dbn_only>::layer_t,
                dll::mp_3d_layer_desc<8, 8, 8, 1, 2, 2, dll::weight_type<float>>::layer_t
            >, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto cdbn = std::make_unique<cdbn_t>();

    cdbn->display();

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    cdbn->pretrain(dataset.training_images, 5);

    dll::dump_timers();
}

DLL_TEST_CASE("perf/kws", "[perf][crbm][mp][cdbn]") {
    using cdbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::conv_rbm_desc<
                    1, 40, 20, 8, 9, 9, dll::weight_type<float>, dll::batch_size<64>, dll::momentum,
                    dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>, dll::shuffle_cond<true>, dll::dbn_only>::layer_t,
                dll::mp_3d_layer_desc<8, 32, 12, 1, 2, 2, dll::weight_type<float>>::layer_t,
                dll::conv_rbm_desc<
                    8, 16, 6, 8, 3, 3, dll::weight_type<float>, dll::batch_size<64>, dll::momentum,
                    dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>, dll::shuffle_cond<true>, dll::dbn_only>::layer_t,
                dll::mp_3d_layer_desc<8, 14, 4, 1, 2, 2, dll::weight_type<float>>::layer_t
            >, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto cdbn = std::make_unique<cdbn_t>();

    cdbn->display();

    auto dataset = mnist::read_dataset<std::vector, std::vector, float>(8192);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    std::vector<etl::fast_dyn_matrix<float, 1, 40, 20>> augmented(dataset.training_images.size());

    for(size_t i = 0; i < dataset.training_images.size(); ++i){
        auto& source_image = dataset.training_images[i];
        auto& target_image = augmented[i];

        for(size_t j = 0; j < source_image.size(); ++j){
            target_image[j] = source_image[j];
        }
    }

    cdbn->pretrain(augmented, 5);

    dll::dump_timers();
}

DLL_TEST_CASE("perf/kws_sub", "[perf][crbm][mp][cdbn]") {
    using cdbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::conv_rbm_desc<
                    1, 40, 20, 8, 9, 9, dll::weight_type<float>, dll::batch_size<64>, dll::momentum,
                    dll::weight_decay<dll::decay_type::L2>, dll::sparsity<dll::sparsity_method::LEE>, dll::shuffle_cond<true>, dll::dbn_only>::layer_t,
                dll::mp_3d_layer_desc<8, 32, 12, 1, 2, 2, dll::weight_type<float>>::layer_t
            >, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto cdbn = std::make_unique<cdbn_t>();

    cdbn->display();

    auto dataset = mnist::read_dataset<std::vector, std::vector, float>(16384);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    std::vector<etl::fast_dyn_matrix<float, 1, 40, 20>> augmented(dataset.training_images.size());

    for(size_t i = 0; i < dataset.training_images.size(); ++i){
        auto& source_image = dataset.training_images[i];
        auto& target_image = augmented[i];

        for(size_t j = 0; j < source_image.size(); ++j){
            target_image[j] = source_image[j];
        }
    }

    cdbn->pretrain(augmented, 5);

    dll::dump_timers();
}
