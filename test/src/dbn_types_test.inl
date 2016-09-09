//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

// Network <- std::vector<float>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/1", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<float>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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

// Network <- std::list<float>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/2", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<float>>(200);
    mnist::binarize_dataset(dataset);

    // mnist reader does not support std::list (with reason)
    std::vector<std::list<float>> training_images;
    training_images.reserve(dataset.training_images.size());
    for(auto& image : dataset.training_images){
        training_images.emplace_back(image.begin(), image.end());
    }

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(training_images, training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(training_images, dataset.training_labels, 50) < 0.9);

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

// Network <- std::deque<float>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/3", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, std::deque<float>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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

// Network <- std::vector<double>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/4", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<double>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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

// Network <- std::list<double>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/5", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<double>>(200);
    mnist::binarize_dataset(dataset);

    // mnist reader does not support std::list (with reason)
    std::vector<std::list<double>> training_images;
    training_images.reserve(dataset.training_images.size());
    for(auto& image : dataset.training_images){
        training_images.emplace_back(image.begin(), image.end());
    }

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(training_images, training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(training_images, dataset.training_labels, 50) < 0.9);

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

// Network <- std::deque<double>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/6", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, std::deque<double>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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

// Network <- etl::dyn_matrix<float, 1>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/7", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/8", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<double, 1>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/9", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/10", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 28 * 28>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/11", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/12", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/13", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/14", "[unit][types]", DBN, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(200);
    mnist::binarize_dataset(dataset);

    typename DBN::dbn_t dbn_fake;
    DBN::init(dbn_fake);

    dbn_fake.pretrain_denoising(dataset.training_images, dataset.training_images, 10);

    typename DBN::dbn_t dbn;
    DBN::init(dbn);

    auto& sample = dataset.training_images[1];

#ifndef TYPES_TEST_NO_PRE
    dbn.pretrain(dataset.training_images, 10);
    dbn.pretrain(dataset.training_images.begin(), dataset.training_images.end(), 10);
#endif
    REQUIRE(dbn.fine_tune(dataset.training_images, dataset.training_labels, 50) < 0.9);

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
