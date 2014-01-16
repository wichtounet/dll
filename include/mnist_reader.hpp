//=======================================================================
// Copyright Baptiste Wicht 2014.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#ifndef DBN_MNIST_READER_HPP
#define DBN_MNIST_READER_HPP

#include <vector>
#include <cstdint>

namespace mnist {

std::vector<std::vector<uint8_t>> read_training_images();
std::vector<std::vector<uint8_t>> read_test_images();

}

#endif
