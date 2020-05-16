//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#define DOCTEST_CONFIG_ASSERTION_PARAMETERS_BY_VALUE
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include <limits>
#include "doctest/doctest.h"

// The second part of the test case is not used with doctest
#define DLL_TEST_CASE(name, description) TEST_CASE(name)

namespace dll_test {

/*!
 * \brief Scale all values of a MNIST dataset into [0,1]
 */
template <typename Dataset>
void mnist_scale(Dataset& dataset) {
    for (auto& image : dataset.training_images) {
        for (auto& pixel : image) {
            pixel *= (1.0 / 256.0);
        }
    }

    for (auto& image : dataset.test_images) {
        for (auto& pixel : image) {
            pixel *= (1.0 / 256.0);
        }
    }
}

} //end of dll_test namespace

#define FT_CHECK(ft_epochs, ft_max)                                                                  \
    {                                                                                                \
        auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, ft_epochs); \
        std::cout << "ft_error:" << ft_error << std::endl;                                           \
        CHECK(ft_error < ft_max);                                                                    \
    }

#define TEST_CHECK(error_max)                                                            \
    {                                                                                    \
        auto test_error = dbn->evaluate_error(dataset.test_images, dataset.test_labels); \
        std::cout << "test_error:" << test_error << std::endl;                           \
        REQUIRE(test_error < error_max);                                                 \
    }

#define FT_CHECK_DATASET_VAL(ft_epochs, ft_max)                                    \
    {                                                                              \
        auto ft_error = dbn->fine_tune_val(dataset.train(), dataset.val(), ft_epochs); \
        std::cout << "ft_error:" << ft_error << std::endl;                         \
        CHECK(ft_error < ft_max);                                                  \
    }

#define FT_CHECK_DATASET(ft_epochs, ft_max)                         \
    {                                                               \
        auto ft_error = dbn->fine_tune(dataset.train(), ft_epochs); \
        std::cout << "ft_error:" << ft_error << std::endl;          \
        CHECK(ft_error < ft_max);                                   \
    }

#define TEST_CHECK_DATASET(error_max)                          \
    {                                                          \
        auto test_error = dbn->evaluate_error(dataset.test()); \
        std::cout << "test_error:" << test_error << std::endl; \
        REQUIRE(test_error < error_max);                       \
    }

#define FT_CHECK_2(net, dataset, ft_epochs, ft_max)                     \
    {                                                                   \
        auto ft_error = net->fine_tune_val(dataset.train(), ft_epochs); \
        std::cout << "ft_error:" << ft_error << std::endl;              \
        CHECK(ft_error < ft_max);                                       \
    }

#define FT_CHECK_2_VAL(net, dataset, ft_epochs, ft_max)                                \
    {                                                                                  \
        auto ft_error = net->fine_tune_val(dataset.train(), dataset.val(), ft_epochs); \
        std::cout << "ft_error:" << ft_error << std::endl;                             \
        CHECK(ft_error < ft_max);                                                      \
    }

#define TEST_CHECK_2(net, dataset, error_max)                  \
    {                                                          \
        auto test_error = net->evaluate_error(dataset.test()); \
        std::cout << "test_error:" << test_error << std::endl; \
        REQUIRE(test_error < error_max);                       \
    }
