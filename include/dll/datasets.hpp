//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/generators.hpp"

namespace dll {

template<typename G>
inline void fill_length(const char* name, std::unique_ptr<G>& generator, std::array<size_t, 4>& column_length){
    if(generator){
        column_length[0] = std::max(column_length[0], std::string(name).size());
        column_length[1] = std::max(column_length[1], std::to_string(generator->size()).size());
        column_length[2] = std::max(column_length[2], std::to_string(generator->batches()).size());
        column_length[3] = std::max(column_length[3], std::to_string(generator->augmented_size()).size());
    }
}

template <>
inline void fill_length([[maybe_unused]] const char* name, [[maybe_unused]] std::unique_ptr<int>& g, [[maybe_unused]] std::array<size_t, 4>& column_length) {}

template<typename G>
inline void print_line(const char* name, std::unique_ptr<G>& generator, std::array<size_t, 4>& column_length){
    if(generator){
        printf(" | %-*s | %-*s | %-*s | %-*s |\n",
            int(column_length[0]), name,
            int(column_length[1]), std::to_string(generator->size()).c_str(),
            int(column_length[2]), std::to_string(generator->batches()).c_str(),
            int(column_length[3]), std::to_string(generator->augmented_size()).c_str());
    }
}

template <>
inline void print_line([[maybe_unused]] const char* name, [[maybe_unused]] std::unique_ptr<int>& g, [[maybe_unused]] std::array<size_t, 4>& column_length) {}

/*!
 * \brief A dataset.
 *
 * A dataset is made of a train data generator and a test data generator.
 * Optionally, a validation data generator is also available.
 */
template<typename TrainG, typename TestG, typename ValG>
struct dataset_holder {
private:
    std::string name; ///< The name of the dataset
    std::unique_ptr<TrainG> train_generator; ///< The train data generator
    std::unique_ptr<TestG> test_generator;   ///< The test data generator
    std::unique_ptr<ValG> val_generator;     ///< The validation data generator

public:
    /*!
     * \brief Construct a new dataset_holder
     * \param train_generator The train data generator
     * \param test_generator The test data generator
     */
    dataset_holder(std::string name, std::unique_ptr<TrainG>& train_generator, std::unique_ptr<TestG>& test_generator)
            : name(std::move(name)), train_generator(std::move(train_generator)), test_generator(std::move(test_generator)) {
        // Nothing else to init
    }

    /*!
     * \brief Construct a new dataset_holder
     * \param train_generator The train data generator
     * \param test_generator The test data generator
     * \param val_generator The validation data generator
     */
    dataset_holder(std::string name, std::unique_ptr<TrainG>& train_generator, std::unique_ptr<TestG>& test_generator, std::unique_ptr<ValG>& val_generator)
            : name(std::move(name)), train_generator(std::move(train_generator)), test_generator(std::move(test_generator)), val_generator(std::move(val_generator)) {
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

    /*!
     * \brief Display information about the dataset on the given stream
     * \param stream The stream to output information to
     * \return stream
     */
    std::ostream& display_pretty(std::ostream& stream){
        constexpr size_t columns = 4;

        std::cout << '\n';

        std::array<std::string, columns> column_name;
        column_name[0] = name;
        column_name[1] = "Size";
        column_name[2] = "Batches";
        column_name[3] = "Augmented Size";

        std::array<size_t, columns> column_length;
        column_length[0] = column_name[0].size();
        column_length[1] = column_name[1].size();
        column_length[2] = column_name[2].size();
        column_length[3] = column_name[3].size();

        fill_length("train", train_generator, column_length);
        fill_length("val", val_generator, column_length);
        fill_length("test", test_generator, column_length);

        const size_t line_length = (columns + 1) * 1 + 2 + (columns - 1) * 2 + std::accumulate(column_length.begin(), column_length.end(), 0);

        std::cout << " " << std::string(line_length, '-') << '\n';

        printf(" | %-*s | %-*s | %-*s | %-*s |\n",
               int(column_length[0]), column_name[0].c_str(),
               int(column_length[1]), column_name[1].c_str(),
               int(column_length[2]), column_name[2].c_str(),
               int(column_length[3]), column_name[3].c_str());

        std::cout << " " << std::string(line_length, '-') << '\n';

        print_line("train", train_generator, column_length);
        print_line("val", val_generator, column_length);
        print_line("test", test_generator, column_length);

        std::cout << " " << std::string(line_length, '-') << '\n';

        return stream;
    }

    /*!
     * \brief Display information about the dataset on the standard output
     */
    void display_pretty(){
        display_pretty(std::cout);
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
dataset_holder<TrainG, TestG, int> make_dataset_holder(const std::string& name, std::unique_ptr<TrainG>&& train_generator, std::unique_ptr<TestG>&& test_generator){
    return {name, train_generator, test_generator};
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
dataset_holder<TrainG, TestG, ValG> make_dataset_holder(const std::string& name, std::unique_ptr<TrainG>&& train_generator, std::unique_ptr<TestG>&& test_generator, std::unique_ptr<ValG>&& val_generator){
    return {name, train_generator, test_generator, val_generator};
}

} // end of namespace dll

#include "datasets/mnist.hpp"
#include "datasets/mnist_ae.hpp"
#include "datasets/cifar.hpp"
