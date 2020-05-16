//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "mnist/mnist_reader.hpp"

namespace dll {

using mnist_example_t = etl::fast_dyn_matrix<float, 1, 28, 28>;
using mnist_example_nc_t = etl::fast_dyn_matrix<float, 28, 28>;

/*!
 * \brief Create a data generator around the MNIST train set
 * \param folder The folder in which the MNIST train files are
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename Example, typename... Parameters>
auto make_mnist_generator_train_impl(const std::string& folder, size_t start, size_t limit, Parameters&&... /*parameters*/){
    // Create examples for the caches
    Example input;
    float label;

    size_t n = 60000 - start;
    size_t m = 0;

    if(limit > 0 && limit < n){
        n = limit;
        m = limit;
    }

    // Prepare the empty generator
    auto generator = prepare_generator(input, label, n, 10, dll::inmemory_data_generator_desc<Parameters..., dll::categorical>{});

    // Read all the necessary images
    if(!mnist::read_mnist_image_file_flat(generator->input_cache, folder + "/train-images-idx3-ubyte", m, start)){
        std::cerr << "Something went wrong, impossible to load MNIST training images" << std::endl;
        return generator;
    }

    // Read all the labels (categorical)
    generator->label_cache = 0;
    if(!mnist::read_mnist_label_file_categorical(generator->label_cache, folder + "/train-labels-idx1-ubyte", m, start)){
        std::cerr << "Something went wrong, impossible to load MNIST training labels" << std::endl;
        return generator;
    }

    // Apply the transformations on the input
    generator->finalize_prepared_data();

    return generator;
}

/*!
 * \brief Create a data generator around the MNIST test set
 * \param folder The folder in which the MNIST test files are
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename Example, typename... Parameters>
auto make_mnist_generator_test_impl(const std::string& folder, size_t start, size_t limit, Parameters&&... /*parameters*/){
    // Create examples for the caches
    Example input;
    float label;

    size_t n = 10000 - start;
    size_t m = 0;

    if(limit > 0 && limit < n){
        n = limit;
        m = limit;
    }

    // Prepare the empty generator
    auto generator = prepare_generator(input, label, n, 10, dll::inmemory_data_generator_desc<Parameters..., dll::categorical>{});

    // Read all the necessary images
    if(!mnist::read_mnist_image_file_flat(generator->input_cache, folder + "/t10k-images-idx3-ubyte", m, start)){
        std::cerr << "Something went wrong, impossible to load MNIST test images" << std::endl;
        return generator;
    }

    // Read all the labels (categorical)
    generator->label_cache = 0;
    if(!mnist::read_mnist_label_file_categorical(generator->label_cache, folder + "/t10k-labels-idx1-ubyte", m, start)){
        std::cerr << "Something went wrong, impossible to load MNIST test labels" << std::endl;
        return generator;
    }

    // Apply the transformations on the input
    generator->finalize_prepared_data();

    return generator;
}

/*!
 * \brief Create a data generator around the MNIST train set.
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_mnist_generator_train(size_t start, size_t limit, Parameters&&... parameters){
    return make_mnist_generator_train_impl<mnist_example_t>("mnist", start, limit, std::forward<Parameters>(parameters)...);
}

/*!
 * \brief Create a data generator around the MNIST test set
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_mnist_generator_test(size_t start, size_t limit, Parameters&&... parameters){
    return make_mnist_generator_test_impl<mnist_example_t>("mnist", start, limit, std::forward<Parameters>(parameters)...);
}

/*!
 * \brief Creates a dataset around MNIST
 * \param folder The folder in which the MNIST files are
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset(const std::string& folder, Parameters&&... parameters){
    return make_dataset_holder(
        "mnist",
        make_mnist_generator_train_impl<mnist_example_t>(folder, 0UL, 60000UL, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test_impl<mnist_example_t>(folder, 0UL, 10000UL, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around MNIST
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset(Parameters&&... parameters){
    return make_dataset_holder(
        "mnist",
        make_mnist_generator_train(0UL, 60000UL, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(0UL, 10000UL, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around MNIST
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset_nc(Parameters&&... parameters){
    return make_dataset_holder(
        "mnist",
        make_mnist_generator_train_impl<mnist_example_nc_t>("mnist", 0UL, 60000UL, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test_impl<mnist_example_nc_t>("mnist", 0UL, 10000UL, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around MNIST
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset_nc_sub(size_t start, size_t limit, Parameters&&... parameters){
    return make_dataset_holder(
        "mnist",
        make_mnist_generator_train_impl<mnist_example_nc_t>("mnist", start, limit, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test_impl<mnist_example_nc_t>("mnist", 0UL, 10000UL, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around a subset of MNIST
 * \param folder The folder in which the MNIST files are
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset_sub(const std::string& folder, size_t start, size_t limit, Parameters&&... parameters){
    return make_dataset_holder(
        "mnist",
        make_mnist_generator_train_impl<mnist_example_t>(folder, start, limit, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test_impl<mnist_example_t>(folder, 0UL, 10000UL, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around a subset of MNIST
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset_sub(size_t start, size_t limit, Parameters&&... parameters){
    return make_dataset_holder(
        "mnist",
        make_mnist_generator_train(start, limit, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(0UL, 10000UL, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset with a validation set
 *
 * Since MNIST does not have a validation set, it is extracted from the training
 * set.
 *
 * \param folder The folder in which the MNIST files are
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 *
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset_val(const std::string& folder, size_t start, size_t middle, size_t limit, Parameters&&... parameters){
    return make_dataset_holder(
        "mnist",
        make_mnist_generator_train_impl<mnist_example_t>(folder, start, middle, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test_impl<mnist_example_t>(folder, 0UL, 10000UL, std::forward<Parameters>(parameters)...),
        make_mnist_generator_train_impl<mnist_example_t>(folder, middle, limit - middle, std::forward<Parameters>(parameters)...)
    );
}

/*!
 * \brief Creates a dataset with a validation our of MNIST.
 *
 * Since MNIST does not have a validation set, it is extracted from the training
 * set.
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset_val(size_t start, size_t middle, size_t limit, Parameters&&... parameters){
    return make_dataset_holder(
        "mnist",
        make_mnist_generator_train(start, middle, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(0UL, 10000UL, std::forward<Parameters>(parameters)...),
        make_mnist_generator_train(middle, limit - middle, std::forward<Parameters>(parameters)...)
    );
}

} // end of namespace dll
