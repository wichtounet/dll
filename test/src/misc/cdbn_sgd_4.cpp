//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/rbm/rbm.hpp"
#include "dll/rbm/conv_rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/transform/scale_layer.hpp"
#include "dll/transform/shape_3d_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("cdbn/sgd/4", "[dbn][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::shape_3d_layer_desc<1, 28, 28>::layer_t,
            dll::scale_layer_desc<1, 256>::layer_t,
            dll::conv_rbm_square_desc<1, 28, 10, 9, dll::momentum, dll::batch_size<10>, dll::weight_type<float>>::layer_t,
            dll::conv_rbm_square_desc<10, 20, 10, 7, dll::momentum, dll::batch_size<10>, dll::weight_type<float>>::layer_t,
            dll::rbm_desc<10 * 14 * 14, 700, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::rbm_desc<700, 10, dll::momentum, dll::batch_size<10>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;
    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}
