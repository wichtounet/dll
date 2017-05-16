//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "catch.hpp"

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
