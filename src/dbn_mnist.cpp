//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "dbn.hpp"
#include "layer.hpp"
#include "mnist_reader.hpp"
#include "image_utils.hpp"

int main(int argc, char* argv[]){
    auto prediction = false;

    if(argc > 1){
        std::string command(argv[1]);

        if(command == "predict"){
            prediction = true;
        }
    }

    dbn::dbn<dbn::layer<28 * 28, 500>, dbn::layer<500, 2000>, dbn::layer<2000, 10>> dbn;

    auto training_images = mnist::read_training_images();

    binarize_each(training_images);

    //TODO Train

    if(prediction){
        //TODO
    }

    return 0;
}
