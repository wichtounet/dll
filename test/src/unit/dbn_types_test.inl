//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

// Network <- etl::dyn_matrix<float, 1>
TEMPLATE_TEST_CASE_2(TYPES_TEST_PREFIX "/types/7", "[unit][types]", DBN, FLOAT_TYPES_TEST_T1, FLOAT_TYPES_TEST_T2) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

#ifndef TYPES_TEST_NO_PRE
    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 5);
#endif

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 5);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 5);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 10) < 0.99);

    auto fa = dbn.features(sample);
    auto fc = dbn.activation_probabilities(sample);
    auto fd = dbn.train_activation_probabilities(sample);
    auto fe = dbn.test_activation_probabilities(sample);
    auto ff = dbn.full_activation_probabilities(sample);

    REQUIRE(dbn.predict(sample) < 10);
    REQUIRE(dbn.predict_label(fa) < 10);
    REQUIRE(dbn.predict_label(fc) < 10);
    REQUIRE(dbn.predict_label(fd) < 10);
    REQUIRE(dbn.predict_label(fe) < 10);
    cpp_unused(ff);
}

// Network <- etl::dyn_matrix<double, 1>
TEMPLATE_TEST_CASE_2(TYPES_TEST_PREFIX "/types/8", "[unit][types]", DBN, DOUBLE_TYPES_TEST_T1, DOUBLE_TYPES_TEST_T2) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<double, 1>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

#ifndef TYPES_TEST_NO_PRE
    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 5);
#endif

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 5);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 5);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 10) < 0.99);

    auto fa = dbn.features(sample);
    auto fc = dbn.activation_probabilities(sample);
    auto fd = dbn.train_activation_probabilities(sample);
    auto fe = dbn.test_activation_probabilities(sample);
    auto ff = dbn.full_activation_probabilities(sample);

    REQUIRE(dbn.predict(sample) < 10);
    REQUIRE(dbn.predict_label(fa) < 10);
    REQUIRE(dbn.predict_label(fc) < 10);
    REQUIRE(dbn.predict_label(fd) < 10);
    REQUIRE(dbn.predict_label(fe) < 10);
    cpp_unused(ff);
}

// Network <- etl::fast_dyn_matrix<float, 28 * 28>
TEMPLATE_TEST_CASE_2(TYPES_TEST_PREFIX "/types/9", "[unit][types]", DBN, FLOAT_TYPES_TEST_T1, FLOAT_TYPES_TEST_T2) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

#ifndef TYPES_TEST_NO_PRE
    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 5);
#endif

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 5);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 5);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 10) < 0.99);

    auto fa = dbn.features(sample);
    auto fc = dbn.activation_probabilities(sample);
    auto fd = dbn.train_activation_probabilities(sample);
    auto fe = dbn.test_activation_probabilities(sample);
    auto ff = dbn.full_activation_probabilities(sample);

    REQUIRE(dbn.predict(sample) < 10);
    REQUIRE(dbn.predict_label(fa) < 10);
    REQUIRE(dbn.predict_label(fc) < 10);
    REQUIRE(dbn.predict_label(fd) < 10);
    REQUIRE(dbn.predict_label(fe) < 10);
    cpp_unused(ff);
}

// Network <- etl::fast_dyn_matrix<double, 28 * 28>
TEMPLATE_TEST_CASE_2(TYPES_TEST_PREFIX "/types/10", "[unit][types]", DBN, DOUBLE_TYPES_TEST_T1, DOUBLE_TYPES_TEST_T2) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 28 * 28>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

#ifndef TYPES_TEST_NO_PRE
    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 5);
#endif

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 5);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 5);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 10) < 0.99);

    auto fa = dbn.features(sample);
    auto fc = dbn.activation_probabilities(sample);
    auto fd = dbn.train_activation_probabilities(sample);
    auto fe = dbn.test_activation_probabilities(sample);
    auto ff = dbn.full_activation_probabilities(sample);

    REQUIRE(dbn.predict(sample) < 10);
    REQUIRE(dbn.predict_label(fa) < 10);
    REQUIRE(dbn.predict_label(fc) < 10);
    REQUIRE(dbn.predict_label(fd) < 10);
    REQUIRE(dbn.predict_label(fe) < 10);
    cpp_unused(ff);
}

// Network <- etl::fast_dyn_matrix<float, 28 * 28>
TEMPLATE_TEST_CASE_2(TYPES_TEST_PREFIX "/types/11", "[unit][types]", DBN, FLOAT_TYPES_TEST_T1, FLOAT_TYPES_TEST_T2) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

#ifndef TYPES_TEST_NO_PRE
    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 5);
#endif

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 5);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 5);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 10) < 0.99);

    auto fa = dbn.features(sample);
    auto fc = dbn.activation_probabilities(sample);
    auto fd = dbn.train_activation_probabilities(sample);
    auto fe = dbn.test_activation_probabilities(sample);
    auto ff = dbn.full_activation_probabilities(sample);

    REQUIRE(dbn.predict(sample) < 10);
    REQUIRE(dbn.predict_label(fa) < 10);
    REQUIRE(dbn.predict_label(fc) < 10);
    REQUIRE(dbn.predict_label(fd) < 10);
    REQUIRE(dbn.predict_label(fe) < 10);
    cpp_unused(ff);
}

// Network <- etl::fast_dyn_matrix<double, 1, 28, 28>
TEMPLATE_TEST_CASE_2(TYPES_TEST_PREFIX "/types/12", "[unit][types]", DBN, DOUBLE_TYPES_TEST_T1, DOUBLE_TYPES_TEST_T2) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

#ifndef TYPES_TEST_NO_PRE
    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 5);
#endif

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 5);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 5);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 10) < 0.99);

    auto fa = dbn.features(sample);
    auto fc = dbn.activation_probabilities(sample);
    auto fd = dbn.train_activation_probabilities(sample);
    auto fe = dbn.test_activation_probabilities(sample);
    auto ff = dbn.full_activation_probabilities(sample);

    REQUIRE(dbn.predict(sample) < 10);
    REQUIRE(dbn.predict_label(fa) < 10);
    REQUIRE(dbn.predict_label(fc) < 10);
    REQUIRE(dbn.predict_label(fd) < 10);
    REQUIRE(dbn.predict_label(fe) < 10);
    cpp_unused(ff);
}

// Network <- etl::fast_dyn_matrix<float, 28 * 28>
TEMPLATE_TEST_CASE_2(TYPES_TEST_PREFIX "/types/13", "[unit][types]", DBN, FLOAT_TYPES_TEST_T1, FLOAT_TYPES_TEST_T2) {
    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

#ifndef TYPES_TEST_NO_PRE
    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 5);
#endif

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 5);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 5);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 10) < 0.99);

    auto fa = dbn.features(sample);
    auto fc = dbn.activation_probabilities(sample);
    auto fd = dbn.train_activation_probabilities(sample);
    auto fe = dbn.test_activation_probabilities(sample);
    auto ff = dbn.full_activation_probabilities(sample);

    REQUIRE(dbn.predict(sample) < 10);
    REQUIRE(dbn.predict_label(fa) < 10);
    REQUIRE(dbn.predict_label(fc) < 10);
    REQUIRE(dbn.predict_label(fd) < 10);
    REQUIRE(dbn.predict_label(fe) < 10);
    cpp_unused(ff);
}

// Network <- etl::fast_dyn_matrix<double, 1, 28, 28>
TEMPLATE_TEST_CASE_2(TYPES_TEST_PREFIX "/types/14", "[unit][types]", DBN, DOUBLE_TYPES_TEST_T1, DOUBLE_TYPES_TEST_T2) {
    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

#ifndef TYPES_TEST_NO_PRE
    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 5);
#endif

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 5);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 5);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 10) < 0.99);

    auto fa = dbn.features(sample);
    auto fc = dbn.activation_probabilities(sample);
    auto fd = dbn.train_activation_probabilities(sample);
    auto fe = dbn.test_activation_probabilities(sample);
    auto ff = dbn.full_activation_probabilities(sample);

    REQUIRE(dbn.predict(sample) < 10);
    REQUIRE(dbn.predict_label(fa) < 10);
    REQUIRE(dbn.predict_label(fc) < 10);
    REQUIRE(dbn.predict_label(fd) < 10);
    REQUIRE(dbn.predict_label(fe) < 10);
    cpp_unused(ff);
}
