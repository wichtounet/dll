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

} // end of namespace dll

#include "datasets/mnist.hpp"
