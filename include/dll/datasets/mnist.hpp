//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "mnist/mnist_reader.hpp"

namespace dll {

/*!
 * \brief Create a data generator around the MNIST train set
 * \param folder The folder in which the MNIST train files are
 * \param limit The limit size (0 = no limit)
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_mnist_generator_train(const std::string& folder, size_t start, size_t limit, size_t batch, Parameters&&... /*parameters*/){
    // Create examples for the caches
    etl::fast_dyn_matrix<float, 1, 28, 28> input;
    float label;

    size_t n = 60000 - start;
    size_t m = 0;

    if(limit > 0 && limit < n){
        n = limit;
        m = limit;
    }

    // Prepare the empty generator
    auto generator = prepare_generator(input, label, n, 10, dll::inmemory_data_generator_desc<Parameters..., dll::categorical>{}, batch);

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
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_mnist_generator_test(const std::string& folder, size_t start, size_t limit, size_t batch, Parameters&&... /*parameters*/){
    // Create examples for the caches
    etl::fast_dyn_matrix<float, 1, 28, 28> input;
    float label;

    size_t n = 10000 - start;
    size_t m = 0;

    if(limit > 0 && limit < n){
        n = limit;
        m = limit;
    }

    // Prepare the empty generator
    auto generator = prepare_generator(input, label, n, 10, dll::inmemory_data_generator_desc<Parameters..., dll::categorical>{}, batch);

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
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_mnist_generator_train(size_t start = 0, size_t limit = 0, size_t batch = 0, Parameters&&... parameters){
    return make_mnist_generator_train("mnist", start, limit, batch, std::forward<Parameters>(parameters)...);
}

/*!
 * \brief Create a data generator around the MNIST test set
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param limit The limit size (0 = no limit)
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_mnist_generator_test(size_t start = 0, size_t limit = 0, size_t batch = 0, Parameters&&... parameters){
    return make_mnist_generator_test("mnist", start, limit, batch, std::forward<Parameters>(parameters)...);
}

/*!
 * \brief Creates a dataset around MNIST
 * \param folder The folder in which the MNIST files are
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset(const std::string& folder, size_t batch = 0, Parameters&&... parameters){
    return make_dataset_holder(
        make_mnist_generator_train(folder, 0UL, 60000UL, batch, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(folder, 0UL, 10000UL, batch, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around MNIST
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset(size_t batch = 0, Parameters&&... parameters){
    return make_dataset_holder(
        make_mnist_generator_train(0UL, 60000UL, batch, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(0UL, 10000UL, batch, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around a subset of MNIST
 * \param folder The folder in which the MNIST files are
 * \param limit The limit size (0 = no limit)
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset_sub(const std::string& folder, size_t start, size_t limit, size_t batch = 0, Parameters&&... parameters){
    return make_dataset_holder(
        make_mnist_generator_train(folder, start, limit, batch, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(0UL, 10000UL, batch, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around a subset of MNIST
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param limit The limit size (0 = no limit)
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset_sub(size_t start, size_t limit, size_t batch = 0, Parameters&&... parameters){
    return make_dataset_holder(
        make_mnist_generator_train(start, limit, batch, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(0UL, 10000UL, batch, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset with a validation set
 *
 * Since MNIST does not have a validation set, it is extracted from the training
 * set.
 *
 * \param folder The folder in which the MNIST files are
 * \param limit The limit size (0 = no limit)
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 *
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset_val(const std::string& folder, size_t start, size_t middle, size_t limit, size_t batch = 0, Parameters&&... parameters){
    return make_dataset_holder(
        make_mnist_generator_train(folder, start, middle, batch, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(0UL, 10000UL, batch, std::forward<Parameters>(parameters)...),
        make_mnist_generator_train(folder, middle, limit - middle, batch, std::forward<Parameters>(parameters)...)
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
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return The MNIST dataset
 */
template<typename... Parameters>
auto make_mnist_dataset_val(size_t start, size_t middle, size_t limit, size_t batch = 0, Parameters&&... parameters){
    return make_dataset_holder(
        make_mnist_generator_train(start, middle, batch, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(0UL, 10000UL, batch, std::forward<Parameters>(parameters)...),
        make_mnist_generator_train(middle, limit - middle, batch, std::forward<Parameters>(parameters)...)
    );
}

} // end of namespace dll
