//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/generators.hpp"

#include "mnist/mnist_reader.hpp"

namespace dll {

/*!
 * \brief Create a data generator around the MNIST train set
 * \param folder The folder in which the MNIST train files are
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_mnist_generator_train(const std::string& folder, size_t batch = 0, Parameters&&... /*parameters*/){
    // Create examples for the caches
    etl::fast_dyn_matrix<float, 1, 28, 28> input;
    float label;

    // Prepare the empty generator
    auto generator = prepare_generator(input, label, 60000, 10, dll::inmemory_data_generator_desc<Parameters..., dll::categorical>{}, batch);

    // Read all the necessary images
    if(!mnist::read_mnist_image_file_flat(generator->input_cache, folder + "/train-images-idx3-ubyte", 0)){
        std::cerr << "Something went wrong, impossible to load MNIST training images" << std::endl;
        return generator;
    }

    // Read all the labels (categorical)
    generator->label_cache = 0;
    if(!mnist::read_mnist_label_file_categorical(generator->label_cache, folder + "/train-labels-idx1-ubyte", 0)){
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
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_mnist_generator_test(const std::string& folder, size_t batch = 0, Parameters&&... /*parameters*/){
    // Create examples for the caches
    etl::fast_dyn_matrix<float, 1, 28, 28> input;
    float label;

    // Prepare the empty generator
    auto generator = prepare_generator(input, label, 10000, 10, dll::inmemory_data_generator_desc<Parameters..., dll::categorical>{}, batch);

    // Read all the necessary images
    if(!mnist::read_mnist_image_file_flat(generator->input_cache, folder + "/t10k-images-idx3-ubyte", 0)){
        std::cerr << "Something went wrong, impossible to load MNIST test images" << std::endl;
        return generator;
    }

    // Read all the labels (categorical)
    generator->label_cache = 0;
    if(!mnist::read_mnist_label_file_categorical(generator->label_cache, folder + "/t10k-labels-idx1-ubyte", 0)){
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
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_mnist_generator_train(size_t batch = 0, Parameters&&... parameters){
    return make_mnist_generator_train("mnist", batch, std::forward<Parameters>(parameters)...);
}

/*!
 * \brief Create a data generator around the MNIST test set
 *
 * The MNIST train files are assumed to be in a mnist sub folder.
 *
 * \param batch The batch size (0 = default from parameters)
 * \param parameters The parameters of the generator
 * \return a unique_ptr around the create generator
 */
template<typename... Parameters>
auto make_mnist_generator_test(size_t batch = 0, Parameters&&... parameters){
    return make_mnist_generator_test("mnist", batch, std::forward<Parameters>(parameters)...);
}

/*!
 * \brief A dataset.
 *
 * A dataset is made of a train data generator and a test data generator.
 */
template<typename TrainG, typename TestG>
struct dataset_holder {
private:
    std::unique_ptr<TrainG> train_generator; ///< The train data generator
    std::unique_ptr<TestG> test_generator;   ///< The test data generator

public:
    /*!
     * \brief Construct a new dataset_holder
     * \param train_generator The train data generator
     * \param test_generator The test data generator
     */
    dataset_holder(std::unique_ptr<TrainG>& train_generator, std::unique_ptr<TestG>& test_generator)
            : train_generator(std::move(train_generator)), test_generator(std::move(test_generator)) {
        // Nothing else to init
    }

    /*!
     * \brief Returns the generator around the train data
     * \return A reference to the generator for the train data
     */
    TrainG& train(){
        return *train_generator;
    }

    /*!
     * \brief Returns the generator around the test data
     * \return A reference to the generator for the test data
     */
    TestG& test(){
        return *test_generator;
    }

    /*!
     * \brief Display information about the dataset on the given stream
     * \param stream The stream to output information to
     * \return stream
     */
    std::ostream& display(std::ostream& stream){
        std::cout << "MNIST Dataset" << std::endl;

        if(train_generator){
            std::cout << "Train: " << *train_generator;
        }

        if(test_generator){
            std::cout << "Test: " << *test_generator;
        }

        return stream;
    }

    /*!
     * \brief Display information about the dataset on the standard output
     */
    void display(){
        display(std::cout);
    }
};

/*!
 * \brief Prints information about the dataset on the given stream
 * \param os The output stream
 * \param dataset The dataset to print information from
 * \return os
 */
template<typename TrainG, typename TestG>
std::ostream& operator<<(std::ostream& os, dataset_holder<TrainG, TestG>& dataset){
    return dataset.display(os);
}

/*!
 * \brief Helper to create a dataset_holder
 * \param train_generator The train data generator
 * \param test_generator The test data generator
 * \return The dataset holder around the two generators
 */
template<typename TrainG, typename TestG>
dataset_holder<TrainG, TestG> make_dataset_holder(std::unique_ptr<TrainG>&& train_generator, std::unique_ptr<TestG>&& test_generator){
    return {train_generator, test_generator};
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
        make_mnist_generator_train(folder, batch, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(folder, batch, std::forward<Parameters>(parameters)...));
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
        make_mnist_generator_train(batch, std::forward<Parameters>(parameters)...),
        make_mnist_generator_test(batch, std::forward<Parameters>(parameters)...));
}

} // end of namespace dll
