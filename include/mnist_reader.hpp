//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_MNIST_READER_HPP
#define DBN_MNIST_READER_HPP

#include <vector>
#include <cstdint>

namespace mnist {

std::vector<std::vector<uint8_t>> read_training_images();
std::vector<std::vector<uint8_t>> read_test_images();

std::vector<uint8_t> read_training_labels();
std::vector<uint8_t> read_test_labels();

}

#endif
