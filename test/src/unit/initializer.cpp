//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/neural/activation/activation_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

DLL_TEST_CASE("initializer/none", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::init_none>, dll::initializer_bias<dll::init_none>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::init_none>, dll::initializer_bias<dll::init_none>, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 0.9);
}

DLL_TEST_CASE("initializer/zero", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::init_zero>, dll::initializer_bias<dll::init_zero>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::init_zero>, dll::initializer_bias<dll::init_zero>, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 0.9);
}

DLL_TEST_CASE("initializer/gaussian", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::init_normal<>>, dll::initializer_bias<dll::init_normal<>>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::init_normal<>>, dll::initializer_bias<dll::init_normal<>>, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    // Gaussian has very large weights and biases, leading to overfitting
    TEST_CHECK(0.4);
}

DLL_TEST_CASE("initializer/small_gaussian", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100,
                dll::initializer<dll::init_normal<constant(0.0), constant(0.1)>>,
                dll::initializer_bias<dll::init_normal<constant(0.0), constant(0.1)>>
            >::layer_t,
            dll::dense_layer_desc<100, 10,
                dll::initializer<dll::init_normal<constant(0.0), constant(0.1)>>,
                dll::initializer_bias<dll::init_normal<constant(0.0), constant(0.1)>>,
                dll::softmax
            >::layer_t
        >,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("initializer/lecun", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::init_lecun>, dll::initializer_bias<dll::init_lecun>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::init_lecun>, dll::initializer_bias<dll::init_lecun>, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("initializer/xavier", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::init_xavier>, dll::initializer_bias<dll::init_xavier>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::init_xavier>, dll::initializer_bias<dll::init_xavier>, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("initializer/xavier_full", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::init_xavier_full>, dll::initializer_bias<dll::init_xavier>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::init_xavier_full>, dll::initializer_bias<dll::init_xavier>, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}

DLL_TEST_CASE("initializer/he", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer<28 * 28, 100, dll::initializer<dll::init_he>, dll::initializer_bias<dll::init_constant<constant(0.1)>>>,
            dll::dense_layer<100,     10,  dll::initializer<dll::init_he>, dll::initializer_bias<dll::init_constant<constant(0.2)>>, dll::softmax>
        >,
        dll::trainer<dll::sgd_trainer>, dll::batch_size<10>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(500);
    REQUIRE(!dataset.training_images.empty());

    dll_test::mnist_scale(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    CHECK(ft_error < 5e-2);

    TEST_CHECK(0.2);
}
