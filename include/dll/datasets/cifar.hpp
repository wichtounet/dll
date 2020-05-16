//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cifar/cifar10_reader.hpp"

namespace dll {

/*!
 * \brief Create a data generator around the CIFAR-10 train set
 * \param folder The folder in which the CIFAR-10 train files are
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_cifar10_generator_train(const std::string& folder, size_t limit, Parameters&&... /*parameters*/){
    // Create examples for the caches
    etl::fast_dyn_matrix<float, 3, 32, 32> input;
    float label;

    size_t n = 50000;
    size_t m = 0;

    if(limit > 0 && limit < n){
        n = limit;
        m = limit;
    }

    // Prepare the empty generator
    auto generator = prepare_generator(input, label, n, 10, dll::inmemory_data_generator_desc<Parameters..., dll::categorical>{});

    generator->label_cache = 0;

    // Read all the necessary images and labels
    cifar::read_training_categorical(folder, m, generator->input_cache, generator->label_cache);

    // Apply the transformations on the input
    generator->finalize_prepared_data();

    return generator;
}

/*!
 * \brief Create a data generator around the CIFAR-10 test set
 * \param folder The folder in which the CIFAR-10 test files are
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_cifar10_generator_test(const std::string& folder, size_t limit, Parameters&&... /*parameters*/){
    // Create examples for the caches
    etl::fast_dyn_matrix<float, 3, 32, 32> input;
    float label;

    size_t n = 10000;
    size_t m = 0;

    if(limit > 0 && limit < n){
        n = limit;
        m = limit;
    }

    // Prepare the empty generator
    auto generator = prepare_generator(input, label, n, 10, dll::inmemory_data_generator_desc<Parameters..., dll::categorical>{});

    generator->label_cache = 0;

    // Read all the necessary images and labels
    cifar::read_test_categorical(folder, m, generator->input_cache, generator->label_cache);

    // Apply the transformations on the input
    generator->finalize_prepared_data();

    return generator;
}

/*!
 * \brief Create a data generator around the CIFAR-10 train set.
 *
 * The CIFAR-10 train files are assumed to be in a cifar-10 sub folder.
 *
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_cifar10_generator_train(size_t limit, Parameters&&... parameters){
    return make_cifar10_generator_train("cifar-10/cifar-10-batches-bin", limit, std::forward<Parameters>(parameters)...);
}

/*!
 * \brief Create a data generator around the CIFAR-10 test set
 *
 * The CIFAR-10 train files are assumed to be in a cifar-10/cifar-10-batches-bin sub folder.
 *
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_cifar10_generator_test(size_t limit, Parameters&&... parameters){
    return make_cifar10_generator_test("cifar-10/cifar-10-batches-bin", limit, std::forward<Parameters>(parameters)...);
}

/*!
 * \brief Creates a dataset around CIFAR-10
 * \param folder The folder in which the CIFAR-10 files are
 * \param parameters The parameters of the generator
 * \return The CIFAR-10 dataset
 */
template<typename... Parameters>
auto make_cifar10_dataset(const std::string& folder, Parameters&&... parameters){
    return make_dataset_holder(
        "cifar",
        make_cifar10_generator_train(folder, 0, std::forward<Parameters>(parameters)...),
        make_cifar10_generator_test(folder, 0, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around CIFAR-10
 *
 * The CIFAR-10 train files are assumed to be in a cifar-10/cifar-10-batches-bin sub folder.
 *
 * \param parameters The parameters of the generator
 * \return The CIFAR-10 dataset
 */
template<typename... Parameters>
auto make_cifar10_dataset(Parameters&&... parameters){
    return make_dataset_holder(
        "cifar",
        make_cifar10_generator_train(0, std::forward<Parameters>(parameters)...),
        make_cifar10_generator_test(0, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around a subset of CIFAR-10
 * \param folder The folder in which the CIFAR-10 files are
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return The CIFAR-10 dataset
 */
template<typename... Parameters>
auto make_cifar10_dataset_sub(const std::string& folder, size_t limit, Parameters&&... parameters){
    return make_dataset_holder(
        "cifar",
        make_cifar10_generator_train(folder, limit, std::forward<Parameters>(parameters)...),
        make_cifar10_generator_test(0, std::forward<Parameters>(parameters)...));
}

/*!
 * \brief Creates a dataset around a subset of CIFAR-10
 *
 * The CIFAR-10 train files are assumed to be in a cifar-10/cifar-10-batches-bin sub folder.
 *
 * \param limit The limit size (0 = no limit)
 * \param parameters The parameters of the generator
 * \return The CIFAR-10 dataset
 */
template<typename... Parameters>
auto make_cifar10_dataset_sub(size_t limit, Parameters&&... parameters){
    return make_dataset_holder(
        "cifar",
        make_cifar10_generator_train(limit, std::forward<Parameters>(parameters)...),
        make_cifar10_generator_test(0, std::forward<Parameters>(parameters)...));
}

} // end of namespace dll
