//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/generators.hpp"

namespace dll {

/*!
 * \brief A dataset.
 *
 * A dataset is made of a train data generator and a test data generator.
 * Optionally, a validation data generator is also available.
 */
template<typename TrainG, typename TestG, typename ValG>
struct dataset_holder {
private:
    std::unique_ptr<TrainG> train_generator; ///< The train data generator
    std::unique_ptr<TestG> test_generator;   ///< The test data generator
    std::unique_ptr<ValG> val_generator;     ///< The validation data generator

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
     * \brief Construct a new dataset_holder
     * \param train_generator The train data generator
     * \param test_generator The test data generator
     * \param val_generator The validation data generator
     */
    dataset_holder(std::unique_ptr<TrainG>& train_generator, std::unique_ptr<TestG>& test_generator, std::unique_ptr<ValG>& val_generator)
            : train_generator(std::move(train_generator)), test_generator(std::move(test_generator)), val_generator(std::move(val_generator)) {
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
     * \brief Returns the generator around the validation data
     * \return A reference to the generator for the validation data
     */
    ValG& val(){
        return *val_generator;
    }

    /*!
     * \brief Display information about the dataset on the given stream
     * \param stream The stream to output information to
     * \return stream
     */
    std::ostream& display(std::ostream& stream){
        std::cout << "Dataset" << std::endl;

        if(train_generator){
            std::cout << "Training: " << *train_generator;
        }

        if(val_generator){
            std::cout << "Validation: " << *val_generator;
        }

        if(test_generator){
            std::cout << "Testing: " << *test_generator;
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
template<typename TrainG, typename TestG, typename ValG>
std::ostream& operator<<(std::ostream& os, dataset_holder<TrainG, TestG, ValG>& dataset){
    return dataset.display(os);
}

/*!
 * \brief Helper to create a dataset_holder
 * \param train_generator The train data generator
 * \param test_generator The test data generator
 * \return The dataset holder around the two generators
 */
template<typename TrainG, typename TestG>
dataset_holder<TrainG, TestG, int> make_dataset_holder(std::unique_ptr<TrainG>&& train_generator, std::unique_ptr<TestG>&& test_generator){
    return {train_generator, test_generator};
}

/*!
 * \brief Helper to create a dataset_holder
 *
 * \param train_generator The train data generator
 * \param test_generator The test data generator
 * \param val_generator The test data generator
 *
 * \return The dataset holder around the three generators
 */
template<typename TrainG, typename TestG, typename ValG>
dataset_holder<TrainG, TestG, ValG> make_dataset_holder(std::unique_ptr<TrainG>&& train_generator, std::unique_ptr<TestG>&& test_generator, std::unique_ptr<ValG>&& val_generator){
    return {train_generator, test_generator, val_generator};
}

} // end of namespace dll

#include "datasets/mnist.hpp"
#include "datasets/mnist_ae.hpp"
#include "datasets/cifar.hpp"
