//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

// Layer <- std::vector<float>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/1", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<float>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- std::list<float>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/2", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<float>>(100);
    mnist::binarize_dataset(dataset);

    // mnist reader does not support std::list (with reason)
    std::vector<std::list<float>> training_images;
    training_images.reserve(dataset.training_images.size());
    for(auto& image : dataset.training_images){
        training_images.emplace_back(image.begin(), image.end());
    }

    auto& sample = training_images[1];

    REQUIRE(rbm.train(training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(training_images, training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- std::deque<float>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/3", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, std::deque<float>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- std::vector<double>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/4", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<double>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- std::list<double>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/5", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, std::vector<double>>(100);
    mnist::binarize_dataset(dataset);

    // mnist reader does not support std::list (with reason)
    std::vector<std::list<double>> training_images;
    training_images.reserve(dataset.training_images.size());
    for(auto& image : dataset.training_images){
        training_images.emplace_back(image.begin(), image.end());
    }

    auto& sample = training_images[1];

    REQUIRE(rbm.train(training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(training_images, training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- std::deque<double>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/6", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, std::deque<double>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- etl::dyn_matrix<float, 1>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/7", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<float, 1>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- etl::fast_dyn_matrix<float, 1>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/8", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 28 * 28>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- etl::dyn_matrix<double, 1>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/9", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_matrix<double, 1>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- etl::fast_dyn_matrix<double, 1>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/10", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 28 * 28>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- etl::fast_dyn_matrix<float, 1, 28, 28>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/11", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- etl::fast_dyn_matrix<double, 1, 28, 28>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/12", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<double, 1, 28, 28>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- etl::dyn_matrix<float, 3>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/13", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<float, 3>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}

// Layer <- etl::dyn_matrix<double, 3>
TEMPLATE_TEST_CASE_4(TYPES_TEST_PREFIX "/types/14", "[unit][types]", RBM, TYPES_TEST_T1, TYPES_TEST_T2, TYPES_TEST_T3, TYPES_TEST_T4) {
    typename RBM::rbm_t rbm;

    RBM::init(rbm);

    auto dataset = mnist::read_dataset_3d<std::vector, etl::dyn_matrix<double, 3>>(100);
    mnist::binarize_dataset(dataset);

    auto& sample = dataset.training_images[1];

    REQUIRE(rbm.train(dataset.training_images, 20) < 0.2);
    REQUIRE(rbm.reconstruction_error(sample) < 0.2);
    REQUIRE(rbm.train_denoising(dataset.training_images, dataset.training_images, 20) < 1.0);

    auto a = rbm.activate_hidden(sample);
    REQUIRE(rbm.free_energy(sample) < 0.0);

    auto b = rbm.features(sample);

#ifdef TYPES_TEST_MP
    auto c = rbm.hidden_features(sample);

    REQUIRE(rbm.energy(sample, c) != 0.0);
    REQUIRE(rbm.energy(sample, c) != 0.0);

    cpp_unused(a);
    cpp_unused(b);
#else
    REQUIRE(rbm.energy(sample, a) != 0.0);
    REQUIRE(rbm.energy(sample, b) != 0.0);
#endif
}
