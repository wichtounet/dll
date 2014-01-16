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
    dbn::rbm<char,char,char> rbm(28 * 28, 60);

    auto training_images = mnist::read_training_images();

    std::cout << "Training set loaded" << std::endl;

    binarize_each(training_images);

    std::cout << "Images binarized" << std::endl;
    std::cout << "Start training..." << std::endl;

    rbm.train(training_images, 25000);

    //TODO

    return 0;
}
