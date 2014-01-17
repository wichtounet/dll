//=======================================================================
// Copyright Baptiste Wicht 2014.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#include <iostream>

#include "rbm.hpp"
#include "mnist_reader.hpp"
#include "image_utils.hpp"

int main(){
    dbn::rbm<uint8_t, uint8_t, 50, true> rbm(28 * 28, 36);

    auto training_images = mnist::read_training_images();

    binarize_each(training_images);

    rbm.train(training_images, 100);

    return 0;
}
