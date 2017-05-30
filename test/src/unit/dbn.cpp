//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "dll_test.hpp"

#define DLL_SVM_SUPPORT

#include "dll/dbn.hpp"
#include "dll/rbm/rbm.hpp"
#include "dll/rbm/dyn_rbm.hpp"
#include "dll/transform/shape_layer_1d.hpp"
#include "dll/transform/binarize_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE("unit/dbn/mnist/1", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 150, dll::momentum, dll::batch_size<10>, dll::init_weights>::layer_t,
            dll::rbm_desc<150, 250, dll::momentum, dll::batch_size<10>>::layer_t,
            dll::rbm_desc<250, 10, dll::momentum, dll::batch_size<10>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<25>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(300);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    using generator_t = dll::inmemory_data_generator_desc<dll::batch_size<10>, dll::big_batch_size<5>, dll::autoencoder >;

    auto generator = dll::make_generator(
        dataset.training_images, dataset.training_images,
        dataset.training_images.size(), generator_t{});

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(*generator, 25);

    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 5);
    std::cout << "error:" << error << std::endl;
    REQUIRE(error < 5e-2);

    TEST_CHECK(0.3);

    dbn->save_features(dataset.training_images[0], ".tmp.features");

    std::ifstream is(".tmp.features");

    size_t big = 0;
    for (size_t i = 0; i < 10; ++i) {
        double v;
        char dump;
        is >> v;

        if (i < 9) {
            is >> dump;
        }

        if (v > 0.01) {
            ++big;
        }
    }

    REQUIRE(big == 1);
}

TEST_CASE("unit/dbn/mnist/2", "[dbn][unit]") {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(250);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    typedef dll::dbn_desc<
        dll::dbn_label_layers<
            dll::rbm_desc<28 * 28, 200, dll::batch_size<50>, dll::init_weights, dll::momentum>::layer_t,
            dll::rbm_desc<200, 300, dll::batch_size<50>, dll::momentum>::layer_t,
            dll::rbm_desc<310, 500, dll::batch_size<50>, dll::momentum>::layer_t>,
        dll::batch_size<10>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_simple_t;

    auto dbn = std::make_unique<dbn_simple_t>();

    dbn->train_with_labels(dataset.training_images, dataset.training_labels, 10, 10);

    auto error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::label_predictor());
    std::cout << "test_error:" << error << std::endl;
    REQUIRE(error < 0.3);
}

TEST_CASE("unit/dbn/mnist/3", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 200, dll::momentum, dll::batch_size<20>, dll::visible<dll::unit_type::GAUSSIAN>>::layer_t,
            dll::rbm_desc<200, 300, dll::momentum, dll::batch_size<20>>::layer_t,
            dll::rbm_desc<300, 10, dll::momentum, dll::batch_size<20>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<10>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);

    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 5);
    REQUIRE(error < 5e-2);

    TEST_CHECK(0.25);
}

TEST_CASE("unit/dbn/mnist/4", "[dbn][cg][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 150, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<150, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_mode, dll::batch_size<25>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    REQUIRE(dbn->batch_mode());

    dbn->pretrain(dataset.training_images, 20);

    auto error = dbn->fine_tune(
        dataset.training_images.begin(), dataset.training_images.end(),
        dataset.training_labels.begin(), dataset.training_labels.end(),
        5);

    REQUIRE(error < 5e-2);

    TEST_CHECK(0.25);

    //Mostly here to ensure compilation
    auto out = dbn->prepare_one_output<etl::dyn_matrix<float, 1>>();
    REQUIRE(out.size() > 0);
}

TEST_CASE("unit/dbn/mnist/5", "[dbn][sgd][unit]") {
    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 150, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<150, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::momentum, dll::batch_size<25>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.05;

    dbn->pretrain(dataset.training_images, 20);

    auto error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << error << std::endl;
    REQUIRE(error < 1e-1);

    TEST_CHECK(0.3);
}

TEST_CASE("unit/dbn/mnist/6", "[dbn][dyn][unit]") {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::dyn_rbm_desc<dll::momentum, dll::init_weights>::layer_t,
                dll::dyn_rbm_desc<dll::momentum>::layer_t,
                dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
            dll::batch_size<25>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().init_layer(28 * 28, 100);
    dbn->template layer_get<1>().init_layer(100, 200);
    dbn->template layer_get<2>().init_layer(200, 10);

    dbn->pretrain(dataset.training_images, 20);

    TEST_CHECK(1.0);
}

TEST_CASE("unit/dbn/mnist/7", "[dbn][svm][unit]") {
    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t>,
        dll::batch_size<25>, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(dataset.training_images, 20);
    auto result = dbn->svm_train(dataset.training_images, dataset.training_labels);

    REQUIRE(result);

    auto test_error = dll::test_set(dbn, dataset.training_images, dataset.training_labels, dll::svm_predictor());
    std::cout << "test_error:" << test_error << std::endl;
    REQUIRE(test_error < 0.2);
}

// Pretrain with binarize layer
TEST_CASE("unit/dbn/mnist/8", "[dbn][unit]") {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::shape_layer_1d_desc<28 * 28>::layer_t,
            dll::binarize_layer_desc<30>::layer_t,
            dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<100, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_size<25>, dll::trainer<dll::cg_trainer>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(250);
    REQUIRE(!dataset.training_images.empty());

    auto dbn = std::make_unique<dbn_t>();
    dbn->pretrain(dataset.training_images, 20);
}

// Pretrain in denoising mode
// Not include in standard test suite (covered by unit/dbn/mnist/10)
TEST_CASE("unit/dbn/mnist/9", "[dbn][denoising][unit_full]") {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::rbm_desc<
                    28 * 28, 200,
                    dll::batch_size<25>,
                    dll::momentum,
                    dll::weight_decay<>,
                    dll::visible<dll::unit_type::GAUSSIAN>,
                    dll::shuffle,
                    dll::weight_type<float>>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().learning_rate *= 5;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(250);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto noisy = dataset.training_images;

    std::default_random_engine rand_engine(56);
    std::normal_distribution<float> normal_distribution(0.0, 0.5);
    auto noise = std::bind(normal_distribution, rand_engine);

    for (auto& image : noisy) {
        for (auto& noisy_x : image) {
            noisy_x += noise();
        }
    }

    mnist::normalize_each(noisy);

    dbn->pretrain_denoising(noisy, dataset.training_images, 50);
}

TEST_CASE("unit/dbn/mnist/10", "[dbn][denoising][unit]") {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::rbm_desc<
                    28 * 28, 200,
                    dll::batch_size<25>,
                    dll::momentum,
                    dll::weight_decay<>,
                    dll::visible<dll::unit_type::GAUSSIAN>,
                    dll::shuffle>::layer_t,
                dll::rbm_desc<
                    200, 200,
                    dll::batch_size<25>,
                    dll::momentum,
                    dll::weight_decay<>,
                    dll::visible<dll::unit_type::BINARY>,
                    dll::shuffle>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().learning_rate *= 5;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(250);
    REQUIRE(!dataset.training_images.empty());

    mnist::normalize_dataset(dataset);

    auto noisy = dataset.training_images;

    std::default_random_engine rand_engine(56);
    std::normal_distribution<float> normal_distribution(0.0, 0.5);
    auto noise = std::bind(normal_distribution, rand_engine);

    for (auto& image : noisy) {
        for (auto& noisy_x : image) {
            noisy_x += noise();
        }
    }

    mnist::normalize_each(noisy);

    dbn->pretrain_denoising(noisy, dataset.training_images, 50);
}

TEST_CASE("unit/dbn/mnist/11", "[dbn][denoising][unit]") {
    using dbn_t =
        dll::dbn_desc<
            dll::dbn_layers<
                dll::rbm_desc<
                    28 * 28, 200,
                    dll::batch_size<25>,
                    dll::momentum,
                    dll::weight_decay<>,
                    dll::visible<dll::unit_type::BINARY>,
                    dll::shuffle>::layer_t,
                dll::rbm_desc<
                    200, 200,
                    dll::batch_size<25>,
                    dll::momentum,
                    dll::weight_decay<>,
                    dll::visible<dll::unit_type::BINARY>,
                    dll::shuffle>::layer_t>, dll::trainer<dll::cg_trainer>>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->template layer_get<0>().learning_rate *= 5;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>(250);
    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    dbn->pretrain_denoising_auto(dataset.training_images, 50, 0.3);
}

// Batch mode
TEST_CASE("unit/dbn/mnist/12", "[dbn][unit]") {
    dll::reset_timers();

    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::rbm_desc<28 * 28, 150, dll::momentum, dll::batch_size<25>, dll::init_weights>::layer_t,
            dll::rbm_desc<150, 200, dll::momentum, dll::batch_size<25>>::layer_t,
            dll::rbm_desc<200, 10, dll::momentum, dll::batch_size<25>, dll::hidden<dll::unit_type::SOFTMAX>>::layer_t>,
        dll::batch_mode, dll::momentum, dll::trainer<dll::sgd_trainer>, dll::batch_size<25>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(250);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    auto dbn = std::make_unique<dbn_t>();

    REQUIRE(dbn->batch_mode());

    dbn->learning_rate = 0.05;

    dbn->pretrain(dataset.training_images, 20);

    auto error = dbn->fine_tune(
        dataset.training_images.begin(), dataset.training_images.end(),
        dataset.training_labels.begin(), dataset.training_labels.end(),
        50);

    REQUIRE(error < 5e-2);

    TEST_CHECK(0.25);

    //Mostly here to ensure compilation
    auto out = dbn->prepare_one_output<etl::dyn_matrix<float, 1>>();
    REQUIRE(out.size() > 0);

    dll::dump_timers();
}
