//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*
 * \file
 * \brief Tests for data augmentations and generators
 */

#include <deque>

#include "dll_test.hpp"

#include "dll/dbn.hpp"
#include "dll/rbm/rbm.hpp"
#include "dll/neural/dense/dense_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

// Use a simple in-memory generator for fine-tuning
DLL_TEST_CASE("unit/augment/mnist/1", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 300>::layer_t,
            dll::dense_layer_desc<300, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    using train_generator_t = dll::inmemory_data_generator_desc<dll::batch_size<25>, dll::categorical, dll::scale_pre<255>>;

    auto train_generator = dll::make_generator(
        dataset.training_images, dataset.training_labels,
        dataset.training_images.size(), 10,
        train_generator_t{});

    auto test_generator = dll::make_generator(
        dataset.test_images, dataset.test_labels,
        dataset.test_images.size(), 10,
        train_generator_t{});

    auto dbn = std::make_unique<dbn_t>();

    auto error = dbn->fine_tune(*train_generator, 50);
    std::cout << "error:" << error << std::endl;
    CHECK(error < 5e-2);

    auto test_error = dbn->evaluate_error(*test_generator);
    std::cout << "test_error:" << test_error << std::endl;
    CHECK(test_error < 0.3);
}

// Use a simple in-memory generator for pretraining and fine-tuning
DLL_TEST_CASE("unit/augment/mnist/2", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::rbm_desc<200, 300, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::rbm_desc<300, 10, dll::momentum, dll::batch_size<10>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    using pretrain_generator_t = dll::inmemory_data_generator_desc<dll::batch_size<10>, dll::autoencoder, dll::binarize_pre<30>>;
    using train_generator_t = dll::inmemory_data_generator_desc<dll::batch_size<25>, dll::categorical, dll::binarize_pre<30>>;

    auto pretrain_generator = dll::make_generator(
        dataset.training_images, dataset.training_images,
        dataset.training_images.size(), 10,
        pretrain_generator_t{});

    auto train_generator = dll::make_generator(
        dataset.training_images, dataset.training_labels,
        dataset.training_images.size(), 10,
        train_generator_t{});

    auto test_generator = dll::make_generator(
        dataset.test_images, dataset.test_labels,
        dataset.test_images.size(), 10,
        train_generator_t{});

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(*pretrain_generator, 25);

    auto error = dbn->fine_tune(*train_generator, 25);
    std::cout << "error:" << error << std::endl;
    REQUIRE(error < 5e-2);

    auto test_error = dbn->evaluate_error(*test_generator);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.3);
}

// Use a simple out-memory generator for fine-tuning
DLL_TEST_CASE("unit/augment/mnist/3", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 300>::layer_t,
            dll::dense_layer_desc<300, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    using train_generator_t = dll::outmemory_data_generator_desc<dll::batch_size<25>, dll::categorical, dll::scale_pre<255>>;

    auto train_generator = dll::make_generator(
        dataset.training_images, dataset.training_labels,
        dataset.training_images.size(), 10,
        train_generator_t{});

    auto test_generator = dll::make_generator(
        dataset.test_images, dataset.test_labels,
        dataset.test_images.size(), 10,
        train_generator_t{});

    auto dbn = std::make_unique<dbn_t>();

    auto error = dbn->fine_tune(*train_generator, 50);
    std::cout << "error:" << error << std::endl;
    CHECK(error < 5e-2);

    auto test_error = dbn->evaluate_error(*test_generator);
    std::cout << "test_error:" << test_error << std::endl;
    CHECK(test_error < 0.3);
}

// Use a simple out-memory generator for pretraining and fine-tuning
DLL_TEST_CASE("unit/augment/mnist/4", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::rbm_desc<200, 300, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::rbm_desc<300, 10, dll::momentum, dll::batch_size<10>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    using pretrain_generator_t = dll::outmemory_data_generator_desc<dll::batch_size<10>, dll::autoencoder, dll::binarize_pre<30>>;
    using train_generator_t = dll::outmemory_data_generator_desc<dll::batch_size<25>, dll::categorical, dll::binarize_pre<30>>;

    auto pretrain_generator = dll::make_generator(
        dataset.training_images, dataset.training_images,
        dataset.training_images.size(), 10,
        pretrain_generator_t{});

    auto train_generator = dll::make_generator(
        dataset.training_images, dataset.training_labels,
        dataset.training_images.size(), 10,
        train_generator_t{});

    auto test_generator = dll::make_generator(
        dataset.test_images, dataset.test_labels,
        dataset.test_images.size(), 10,
        train_generator_t{});

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(*pretrain_generator, 25);

    auto error = dbn->fine_tune(*train_generator, 25);
    std::cout << "error:" << error << std::endl;
    REQUIRE(error < 5e-2);

    auto test_error = dbn->evaluate_error(*test_generator);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.3);
}

// Use a simple in-memory generator for fine-tuning with augmentation
DLL_TEST_CASE("unit/augment/mnist/5", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 300>::layer_t,
            dll::dense_layer_desc<300, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::batch_size<20>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(600);
    REQUIRE(!dataset.training_images.empty());

    using train_generator_t = dll::inmemory_data_generator_desc<dll::batch_size<20>, dll::noise<20>, dll::categorical, dll::scale_pre<255>>;

    auto train_generator = dll::make_generator(
        dataset.training_images, dataset.training_labels,
        dataset.training_images.size(), 10,
        train_generator_t{});

    auto test_generator = dll::make_generator(
        dataset.test_images, dataset.test_labels,
        dataset.test_images.size(), 10,
        train_generator_t{});

    auto dbn = std::make_unique<dbn_t>();

    auto error = dbn->fine_tune(*train_generator, 60);
    std::cout << "error:" << error << std::endl;
    CHECK(error < 5e-2);

    auto test_error = dbn->evaluate_error(*test_generator);
    std::cout << "test_error:" << test_error << std::endl;
    CHECK(test_error < 0.3);
}

// Use a simple in-memory generator for pretraining and fine-tuning with augmentation
DLL_TEST_CASE("unit/augment/mnist/6", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::rbm_desc<200, 300, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::rbm_desc<300, 10, dll::momentum, dll::batch_size<10>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    using pretrain_generator_t = dll::inmemory_data_generator_desc<dll::batch_size<10>, dll::noise<20>, dll::autoencoder, dll::binarize_pre<30>>;
    using train_generator_t = dll::inmemory_data_generator_desc<dll::batch_size<25>, dll::noise<20>, dll::categorical, dll::binarize_pre<30>>;

    auto pretrain_generator = dll::make_generator(
        dataset.training_images, dataset.training_images,
        dataset.training_images.size(), 10,
        pretrain_generator_t{});

    auto train_generator = dll::make_generator(
        dataset.training_images, dataset.training_labels,
        dataset.training_images.size(), 10,
        train_generator_t{});

    auto test_generator = dll::make_generator(
        dataset.test_images, dataset.test_labels,
        dataset.test_images.size(), 10,
        train_generator_t{});

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(*pretrain_generator, 25);

    auto error = dbn->fine_tune(*train_generator, 25);
    std::cout << "error:" << error << std::endl;
    REQUIRE(error < 5e-2);

    auto test_error = dbn->evaluate_error(*test_generator);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.3);
}

// Use a simple out-memory generator for fine-tuning with augmentation
DLL_TEST_CASE("unit/augment/mnist/7", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_layer_desc<28 * 28, 300>::layer_t,
            dll::dense_layer_desc<300, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    using train_generator_t = dll::outmemory_data_generator_desc<dll::batch_size<25>, dll::noise<20>, dll::categorical, dll::scale_pre<255>>;

    auto train_generator = dll::make_generator(
        dataset.training_images, dataset.training_labels,
        dataset.training_images.size(), 10,
        train_generator_t{});

    auto test_generator = dll::make_generator(
        dataset.test_images, dataset.test_labels,
        dataset.test_images.size(), 10,
        train_generator_t{});

    auto dbn = std::make_unique<dbn_t>();

    auto error = dbn->fine_tune(*train_generator, 50);
    std::cout << "error:" << error << std::endl;
    CHECK(error < 5e-2);

    auto test_error = dbn->evaluate_error(*test_generator);
    std::cout << "test_error:" << test_error << std::endl;
    CHECK(test_error < 0.3);
}

// Use a simple out-memory generator for pretraining and fine-tuning with augmentation
DLL_TEST_CASE("unit/augment/mnist/8", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::rbm_desc<200, 300, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::rbm_desc<300, 10, dll::momentum, dll::batch_size<10>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);
    REQUIRE(!dataset.training_images.empty());

    using pretrain_generator_t = dll::outmemory_data_generator_desc<dll::batch_size<10>, dll::noise<20>, dll::autoencoder, dll::binarize_pre<30>>;
    using train_generator_t = dll::outmemory_data_generator_desc<dll::batch_size<25>, dll::noise<20>, dll::categorical, dll::binarize_pre<30>>;

    auto pretrain_generator = dll::make_generator(
        dataset.training_images, dataset.training_images,
        dataset.training_images.size(), 10,
        pretrain_generator_t{});

    auto train_generator = dll::make_generator(
        dataset.training_images, dataset.training_labels,
        dataset.training_images.size(), 10,
        train_generator_t{});

    auto test_generator = dll::make_generator(
        dataset.test_images, dataset.test_labels,
        dataset.test_images.size(), 10,
        train_generator_t{});

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(*pretrain_generator, 25);

    auto error = dbn->fine_tune(*train_generator, 25);
    std::cout << "error:" << error << std::endl;
    REQUIRE(error < 5e-2);

    auto test_error = dbn->evaluate_error(*test_generator);
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.3);
}
