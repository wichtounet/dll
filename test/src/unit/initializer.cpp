//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#include "dll/neural/dense_layer.hpp"
#include "dll/neural/activation_layer.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("initializer/none", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::initializer_type::NONE>, dll::initializer_bias<dll::initializer_type::NONE>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::initializer_type::NONE>, dll::initializer_bias<dll::initializer_type::NONE>, dll::activation<dll::function::SOFTMAX>>::layer_t
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

TEST_CASE("initializer/zero", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::initializer_type::ZERO>, dll::initializer_bias<dll::initializer_type::ZERO>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::initializer_type::ZERO>, dll::initializer_bias<dll::initializer_type::ZERO>, dll::activation<dll::function::SOFTMAX>>::layer_t
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

TEST_CASE("initializer/gaussian", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::initializer_type::GAUSSIAN>, dll::initializer_bias<dll::initializer_type::GAUSSIAN>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::initializer_type::GAUSSIAN>, dll::initializer_bias<dll::initializer_type::GAUSSIAN>, dll::activation<dll::function::SOFTMAX>>::layer_t
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

TEST_CASE("initializer/small_gaussian", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::initializer_type::SMALL_GAUSSIAN>, dll::initializer_bias<dll::initializer_type::SMALL_GAUSSIAN>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::initializer_type::SMALL_GAUSSIAN>, dll::initializer_bias<dll::initializer_type::SMALL_GAUSSIAN>, dll::activation<dll::function::SOFTMAX>>::layer_t
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

TEST_CASE("initializer/lecun", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::initializer_type::LECUN>, dll::initializer_bias<dll::initializer_type::LECUN>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::initializer_type::LECUN>, dll::initializer_bias<dll::initializer_type::LECUN>, dll::activation<dll::function::SOFTMAX>>::layer_t
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

TEST_CASE("initializer/xavier", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::initializer_type::XAVIER>, dll::initializer_bias<dll::initializer_type::XAVIER>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::initializer_type::XAVIER>, dll::initializer_bias<dll::initializer_type::XAVIER>, dll::activation<dll::function::SOFTMAX>>::layer_t
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

TEST_CASE("initializer/xavier_full", "[dense][unit][mnist][sgd]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 100, dll::initializer<dll::initializer_type::XAVIER_FULL>, dll::initializer_bias<dll::initializer_type::XAVIER>>::layer_t,
            dll::dense_layer_desc<100, 10, dll::initializer<dll::initializer_type::XAVIER_FULL>, dll::initializer_bias<dll::initializer_type::XAVIER>, dll::activation<dll::function::SOFTMAX>>::layer_t
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
